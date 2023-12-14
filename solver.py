import datetime
import os
import sys
import time

import numpy as np
import ruamel.yaml
import torch.distributed as dist
import yaml
from torch.cuda import amp
from tqdm import tqdm
import torch

# loss function for yolov4csp
from vitg.checkpoint.checkpoint_main import Checkpoint
from vitg.metric.metric_main import calculate_train_map
from vitg.network.backbone.vitgyolor.utils.loss import (
    compute_loss as compute_loss_yolor,
)
from vitg.logger.log import ShowProgressBar
from vitg.network.backbone.vitgyolov7.utils.loss import ComputeLoss as ComputeLossv7
from vitg.network.backbone.vitgyolov8 import YOLO as yolov8orimodel
from vitg.output.output_main import Output
from vitg.symbols.extras import (
    EarlyStopping,
    Force_close,
    Lossv8func,
    save_c_model_val2,
)

# dataloader
from vitg.utils.datasets import create_dataloader, create_dataloader_noletterbox

# other functions for test
from vitg.utils.general import (
    ap_per_class,
    check_img_size,
    clip_coords,
    non_max_suppression,
    xywh2xyxy,
)
from vitg.utils.helper import get_cateDict_classNum, modified_parameters

# from vitg.utils.layers import *
from vitg.utils.torch_utils import select_device, time_synchronized

yamldy = ruamel.yaml.YAML()

from contextlib import suppress

from torchvision.ops.boxes import box_iou

from vitg.loader.loader_main import (
    get_intertrain_loader,
    get_test_loader,
    get_train_loader,
)

# ComputeLoss
from vitg.loss.loss_main import Loss
from vitg.network.backbone.mobilenetSSD.src.loss import Loss as SSDLoss
from vitg.network.backbone.mobilenetSSD.src.process import (
    evaluate_lmdb_outyaml_mobilenet,
    evaluate_mobilenet,
)

# mobilenet SSD
from vitg.network.backbone.mobilenetSSD.src.utils import Encoder

# yolov8
from vitg.network.backbone.vitgyolov8.yolo.utils import ops as yolov8ops
from vitg.network.backbone.vitgyolov8.yolo.utils import yaml_load
from vitg.network.backbone.vitgyolov8.yolo.utils.torch_utils import de_parallel
from vitg.network.network_main import Network
from vitg.symbols.loss import compute_loss
from vitg.trainer.trainer_main import BaseTrainer

# Constants
from vitg.utils.helper import get_hyp, get_optimizer_scheduler
from vitg.visualization.visualize import Visualize


class Solver(BaseTrainer):
    """
    Solver for training and testing
    """

    def __init__(self, config):
        super().__init__()
        get_cateDict_classNum(config)

        modified_parameters(config)

        self.config = config

        self.device = select_device(config.device, batch_size=config.batch_size)
        self.hyp = get_hyp(config)

        self.amp_autocast = torch.cuda.amp.autocast if self.config.use_amp else suppress

        if self.config.use_tensorboard:
            self.visualize = Visualize(visualize_type="tensorboard")
        else:
            self.visualize = None
        from vitg.logger.log import get_logger

        self.logger = get_logger(config.log_path)

        self.output = Output(config)

    def training_step(self, rank, dataloader, epoch, forceclose_check):
        self.net.train()

        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            self.progress_cls = ShowProgressBar(self.config.arch, self.device)
            pbar = tqdm(pbar, total=self.nb)  # progress bar
        self.optimizer.zero_grad()
        for i, (
            imgs,
            targets,
            _,
            _,
        ) in pbar:
            if forceclose_check.check_close():
                print("detect forceclose in train bach")
                if epoch == 0:
                    forceclose_check.reset_forceclose()
                    print(
                        "Since the training is not completed, reseting the forceclose status..."
                    )
                else:
                    break

            gloc, glabel = self.get_gloc_glabel(targets, self.net)

            self.ni = (
                i + self.nb * epoch
            )  # number integrated batches (since train start)
            imgs = (
                imgs.to(self.device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            self.run_warmup(self.ni, self.nw, epoch)

            # Forward pass
            loss, loss_items = self.run_forward(
                imgs, targets, self.net, gloc, glabel, self.loss_fun, rank
            )

            # Backward pass
            self.run_backward(loss, self.scaler)

            # Optimizer Steps
            self.optimizer_step(self.scaler, self.net.model)

            # Update progress bar
            s = self.progress_cls(
                (epoch, self.config.epochs),
                [loss, loss_items],
                [imgs, targets],
            )

            pbar.set_description(s)

        # Scheduler
        self.scheduler.step()

    def train(self):
        ckpt_mgt = Checkpoint(self.config)

        best_val = os.path.join(
            self.config.model_dir, f"{self.config.name}model_val.pth"
        )
        last_checkpoint = os.path.abspath(self.config.checkpoint_dir)
        resume_train, weights = ckpt_mgt.load_checkpoint(last_checkpoint, self.device)

        # Model checkpoint
        self.net = Network(self.config, self.hyp, self.device, resume_train)
        model = self.net.model

        rank = self.config.global_rank
        self.config.weights = weights
        cuda = self.device.type != "cpu"
        self.optimizer, self.lf, self.scheduler = get_optimizer_scheduler(
            self.hyp, self.config.epochs, self.net
        )
        # Resume
        start_epoch, best_train_map, best_val_map = ckpt_mgt.check_retrain(
            self.config.epochs,
            self.config.weights,
            self.net.exclude,
            resume_train,
            self.net.model,
            self.optimizer,
        )

        # Image sizes
        gs = self.config.gs  # grid size (max stride)
        imgsz, imgsz_test = [
            check_img_size(x, gs) for x in self.config.img_size
        ]  # verify imgsz are gs-multiples

        # DP mode
        self.config.imgsz = imgsz
        self.config.imgsz_test = imgsz_test
        self.config.gs = gs
        self.config.rank = rank
        
        if cuda and rank == -1 and torch.cuda.device_count() > 1:
            self.net.model = torch.nn.DataParallel(self.net.model)

        # use_rec_train= not self.config.mosaic
        

        # Trainloader
        dataloader = get_train_loader(
            self.config,
            self.hyp,
        )

        self.nb = len(dataloader)  # number of batches

        # Testloader
        if rank in [-1, 0]:
            # local_rank is set to -1. Because only the first process is expected to do evaluation.
            (
                testloader_noaug,
                trainloader_noaug,
                testloader_train_yml,
                testloader_val_yml,
            ) = get_test_loader(
                self.config, self.hyp
            )

            # ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
            # local_rank is set to -1. Because only the first process is expected to do evaluation.

            # train metric
            train_loader_eval = get_intertrain_loader(
                self.config, self.hyp
            )

        # Model parameters
        self.hyp["cls"] *= (
            self.net.nc / 80.0
        )  # scale coco-tuned hyp['cls'] to current dataset
        model.nc = self.net.nc  # attach number of classes to model
        model.hyp = self.hyp  # attach hyperparameters to model
        model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
        model.names = self.net.names

        self.nw = max(
            3 * self.nb, 1e3
        )  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        maps = np.zeros(self.net.nc)  # mAP per class
        self.scheduler.last_epoch = start_epoch - 1  # do not move

        # add mix precision
        self.scaler = amp.GradScaler(enabled=cuda) if self.config.use_amp else None

        if rank in [0, -1]:
            print("Image sizes %g train, %g test" % (imgsz, imgsz_test))
            print("Using %g dataloader workers" % dataloader.num_workers)
            print("Starting training for %g epochs..." % self.config.epochs)

        # initialize the early_stopping object
        early_stoppingvade = EarlyStopping(
            patience=self.config.early_stop, verbose=True
        )

        forceclose_check = Force_close()

        self.loss_fun = Loss(self.config, model, self.net.dboxes, self.net.device)
        # encoder = loss_fun.encoder
        # criterion = loss_fun.criterion
        # t0_log = time.time()
        # dataloader variable name: usage
        # train_loader_eval       : train, metric compute [except mobilenetssd]
        # dataloader              : train, augmentation [all archs]

        # testloader_train_yml    : train, np augmentation [except mobilenetssd]
        # testloader_val_yml      : test, np augmentation [except mobilenetssd]
        # testloader_noaug        : test, no augmentation [mobilenetssd]
        # trainloader_noaug       : train, no augmentation [mobilenetssd]
        # =================== Main Training Looop===================
        for counterj, epoch in enumerate(range(start_epoch, self.config.epochs)):
            # Start training
            t0 = time.time()

            self.training_step(rank, dataloader, epoch, forceclose_check)

            # DDP process 0 or single-GPU
            if rank in [-1, 0]:
                # mAP
                # if ema is not None:
                #     ema.update_attr(model)
                final_epoch = epoch + 1 == self.config.epochs
                self.config.final_epoch = final_epoch

                if not forceclose_check.check_close():
                    if self.config.arch != "mobilenetssd":
                        # if not self.config.notest or final_epoch:  # Calculate mAP for val dataset
                        results, maps, times = self.test(
                            self.config.val_dataset,
                            batch_size=self.config.batch_size,
                            imgsz=imgsz_test,
                            save_json=False,
                            model=model,
                            single_cls=self.config.single_cls,
                            dataloader=testloader_noaug,
                            class_num=self.config.class_number,
                        )

                        end_time = time.time() - t0
                        elapsed_time_r = str(datetime.timedelta(seconds=end_time))[:-7]

                        # Calculate mAP for training dataset
                        (
                            results_train,
                            maps_train,
                            times_train,
                        ) = calculate_train_map(
                            self.config.train_dataset,
                            batch_size=self.config.batch_size,
                            imgsz=imgsz_test,
                            save_json=False,
                            model=model,
                            single_cls=self.config.single_cls,
                            dataloader=train_loader_eval,
                            class_num=self.config.class_number,
                            config=self.config,
                        )

                        train_loss_log = float(self.progress_cls.mloss[-1].item())
                        train_map_log = float(results_train[2])
                        val_loss_log = float(results[4] + results[5] + results[6])
                        val_map_log = float(results[2])
                    else:
                        val_loss, valmap = evaluate_mobilenet(
                            self.config,
                            model,
                            testloader_noaug,
                            epoch,
                            None,
                            self.loss_fun.encoder,
                            0.5,
                            self.loss_fun.criterion,
                            dboxes_default=self.net.dboxes(order="ltrb"),
                        )

                        (
                            trainloss_log_f,
                            trainmap,
                        ) = evaluate_mobilenet(
                            self.config,
                            model,
                            trainloader_noaug,
                            epoch,
                            None,
                            self.loss_fun.encoder,
                            0.5,
                            self.loss_fun.criterion,
                            dboxes_default=self.net.dboxes(order="ltrb"),
                        )
                        end_time = time.time() - t0
                        elapsed_time_r = str(datetime.timedelta(seconds=end_time))[:-7]
                        # end_time_log = time.time() - t0_log
                        train_loss_log = float(trainloss_log_f)
                        train_map_log = float(trainmap)
                        val_loss_log = float(val_loss)
                        val_map_log = float(valmap)

                    # move logger into save_train_log
                    self.logger.info(
                        "Epoch: {} - Elapsed: {}, TrainLoss: {:0.4f}, TrainMap: {:.4f}, ValLoss: {:.4f}, ValMap: {:0.4f}".format(
                            epoch,
                            elapsed_time_r,
                            train_loss_log,
                            train_map_log,
                            val_loss_log,
                            val_map_log,
                        )
                    )

                    self.output.save_training_log(
                        self.config,
                        self.config.out_dir,
                        counterj,
                        epoch,
                        elapsed_time_r,
                        train_loss_log,
                        train_map_log,
                        val_loss_log,
                        val_map_log,
                    )

                if self.visualize:
                    self.visualize.update(epoch, results, self.progress_cls.mloss)

                best_val_map = max(val_map_log, best_val_map)
                best_train_map = max(train_map_log, best_train_map)

                # Save model
                ckpt = {
                    "epoch": epoch,
                    "best_val_map": best_val_map,
                    "best_train_map": best_train_map,
                    "model": self.net.model.state_dict(),
                    "self.optimizer": None
                    if final_epoch
                    else self.optimizer.state_dict(),
                }

                ckpt_mgt.save_checkpoint(last_checkpoint, ckpt, epoch)
                del ckpt

                if best_val_map == val_map_log:
                    best_val_model = self.net.model
                    self.logger.info("save best val epoch model")
                    ckpt = {
                        "epoch": epoch,
                        "best_val_map": best_val_map,
                        "best_train_map": best_train_map,
                        "model": best_val_model.state_dict(),
                        "self.optimizer": None
                        if final_epoch
                        else self.optimizer.state_dict(),
                    }

                    torch.save(ckpt, best_val)
                    save_c_model_val2(self.config, best_val)

                # early stopping
                early_stoppingvade(val_loss_log)
                self.logger.info(f"val loss: {val_loss_log:.4f}")

                if (
                    self.config.final_epoch
                    or early_stoppingvade.early_stop
                    or forceclose_check.check_close()
                ):  # print("active by forceclose")
                    if self.config.arch != "mobilenetssd":
                        # generate val_out.yml when val result improve
                        self.output.generate_val_out_yaml(
                            self.config.val_dataset,
                            batch_size=self.config.yaml_batch_size,
                            imgsz=imgsz_test,
                            save_json=False,
                            model=best_val_model,
                            weights=self.config.weights,
                            single_cls=self.config.single_cls,
                            dataloader=testloader_val_yml,
                            class_num=self.config.class_number,
                            config=self.config,
                        )
                        # generate train_out.yml at final epoch
                        self.output.generate_train_out_yaml(
                            batch_size=self.config.yaml_batch_size,
                            imgsz=imgsz_test,
                            save_json=False,
                            # model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                            model=model,
                            weights=self.config.weights,
                            single_cls=self.config.single_cls,
                            dataloader=testloader_train_yml,
                            class_num=self.config.class_number,
                            config=self.config,
                        )
                        # del ema
                    else:
                        print("start save out yaml")
                        self.config.yaml_out_path = os.path.join(
                            self.config.out_dir,
                            "result",
                            f"{self.config.name}val_out.yaml",
                        )
                        evaluate_lmdb_outyaml_mobilenet(
                            self.config,
                            best_val_model,
                            testloader_noaug,
                            epoch,
                            None,
                            self.loss_fun.encoder,
                            0.5,
                            dboxes_default=self.net.dboxes(order="ltrb"),
                        )
                        print("save val out yaml done")
                        del best_val_model
                        self.config.yaml_out_path = os.path.join(
                            self.config.out_dir,
                            "result",
                            f"{self.config.name}train_out.yaml",
                        )
                        evaluate_lmdb_outyaml_mobilenet(
                            self.config,
                            model,
                            dataloader,
                            epoch,
                            None,
                            self.loss_fun.encoder,
                            0.5,
                            dboxes_default=self.net.dboxes(order="ltrb"),
                        )
                        print("save train out yaml done")
                torch.cuda.empty_cache()

            self.output.save_summary(
                self.config, epoch + 1, best_train_map, best_val_map
            )

            if early_stoppingvade.early_stop:
                print("Early stopping")
                break
            if forceclose_check.check_close():
                print("forceclose in epoch")
                sys.exit(1)

            # end training
            print(
                "%g epochs completed in %.3f hours.\n"
                % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
            )

        dist.destroy_process_group() if rank not in [-1, 0] else None
        torch.cuda.empty_cache()

    def test(
        self,
        data,
        weights=None,
        batch_size=16,
        imgsz=640,
        conf_thres=0.05,
        iou_thres=0.6,  # for NMS
        save_json=False,
        single_cls=False,
        augment=False,
        verbose=False,
        model=None,
        dataloader=None,
        class_num=1,
        merge=False,
    ):
        if self.config.mode == "train":  # called by train.py
            device = next(model.parameters()).device  # get model device
        else:  # called directly
            device = select_device(self.config.device, batch_size=batch_size)

            self.net = Network(self.config, None, device, False)
            model = self.net.model
            if self.config.arch != "mobilenetssd":
                imgsz = check_img_size(imgsz[0], s=64)  # check img_size

            self.net.eval()

        # Half
        half = self.config.use_amp
        if half:
            model.half()

        # Configure
        model.eval()

        nc = class_num
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        # Dataloader
        if self.config.mode != "train":
            if self.config.arch == "mobilenetssd":
                encoder = Encoder(self.net.dboxes)
                criterion = SSDLoss(self.net.dboxes)
                # local_rank is set to -1. Because only the first process is expected to do evaluation.
                dataloader = create_dataloader_noletterbox(
                    self.config.val_dataset,
                    300,
                    batch_size,
                    0,
                    self.config,
                    hyp=None,
                    augment=False,
                    cache=False,
                    pad=0,
                    rect=False,
                )[0]
                if self.config.mode == "val":
                    val_loss, valmap = evaluate_lmdb_outyaml_mobilenet(
                        self.config,
                        model,
                        dataloader,
                        1,
                        None,
                        encoder,
                        0.5,
                        criterion,
                        dboxes_default=self.net.dboxes(order="ltrb"),
                    )
                    self.config.yaml_out_path = os.path.join(
                        self.config.out_dir,
                        "result",
                        f"{self.config.name}val_out.yaml",
                    )
                    self.logger.info("save out yaml to path")
                    self.logger.info(self.config.yaml_out_path)
                    evaluate_mobilenet(
                        self.config,
                        model,
                        dataloader,
                        1,
                        None,
                        encoder,
                        0.5,
                        dboxes_default=self.net.dboxes(order="ltrb"),
                    )
                    return valmap, 0, 0

                elif self.config.mode == "test":
                    self.config.yaml_out_path = os.path.join(
                        self.config.out_dir,
                        "result",
                        self.config.name + "test_out.yaml",
                    )
                    self.logger.info("save out yaml to path")
                    self.logger.info(self.config.yaml_out_path)
                    evaluate_lmdb_outyaml_mobilenet(
                        self.config,
                        model,
                        dataloader,
                        1,
                        None,
                        encoder,
                        0.5,
                        dboxes_default=self.net.dboxes(order="ltrb"),
                    )
                    return None
            elif self.config.arch == "yolov8":
                img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
                _ = (
                    model(img.half() if half else img) if device.type != "cpu" else None
                )  # run once
                dataloader = create_dataloader_noletterbox(
                    self.config.val_dataset,
                    imgsz,
                    batch_size,
                    0,
                    self.config,
                    hyp=None,
                    augment=False,
                    cache=False,
                    pad=0,
                    rect=False,
                )[0]
            else:
                img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
                _ = (
                    model(img.half() if half else img) if device.type != "cpu" else None
                )  # run once
                dataloader = create_dataloader(
                    data,
                    imgsz,
                    batch_size,
                    self.config.gs,
                    self.config,
                    hyp=None,
                    augment=False,
                    cache=False,
                    pad=0,
                    rect=True,
                )[0]

        seen = 0
        names = [str(item) for item in range(nc)]
        s = ("%20s" + "%12s" * 6) % (
            "Class",
            "Images",
            "Targets",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )
        p, r, f1, mp, mr, map50, map, t0, t1 = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        loss = torch.zeros(3, device=device)
        _, stats, ap, ap_class = [], [], [], []

        if self.config.arch == "yolov7" and self.config.mode == "train":
            compute_lossv7 = ComputeLossv7(model)
        elif self.config.arch == "yolov8":
            cfg_dictinterv8dumpy = yaml_load(
                self.config.cfg, append_filename=True
            )  # model dict
            cfg_dictinterv8dumpy["nc"] = self.config.class_number
            interyamlsave = self.config.cfg.replace(".yaml", "_interv8.yaml")

            with open(interyamlsave, "w") as file:
                _ = yaml.dump(cfg_dictinterv8dumpy, file)
            model_v8_loss = yolov8orimodel(interyamlsave)
            model_v8_loss.model.to(device)

            compute_lossv8 = Lossv8func(
                de_parallel(model_v8_loss.model), self.config.class_number
            )

        for img, targets, paths, shapes in tqdm(dataloader, desc=s):
            # print(paths)

            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)

            # Disable gradients
            with torch.no_grad():
                # Run model
                t = time_synchronized()

                if self.config.arch != "yolov8":
                    inf_out, train_out = model(img, augment=augment)
                elif self.config.arch == "yolov8":
                    inf_out = model(img)

                t0 += time_synchronized() - t

                # Compute loss
                if self.config.mode == "train":  # if model has loss hyperparameters
                    if self.config.arch == "yolov7":
                        loss += compute_lossv7([x.float() for x in train_out], targets)[
                            1
                        ][
                            :3
                        ]  # box, obj, cls
                    elif self.config.arch == "yolor":
                        loss += compute_loss_yolor(
                            [x.float() for x in train_out], targets, model
                        )[1][
                            :3
                        ]  # GIoU, obj, cls
                    elif self.config.arch == "yolov4csp":
                        loss += compute_loss(
                            [x.float() for x in train_out], targets, model
                        )[1][
                            :3
                        ]  # GIoU, obj, cls

                    elif self.config.arch == "yolov8":
                        _, loss_items = compute_lossv8(inf_out[1], targets)

                        loss += loss_items.detach().to(device)

                # Run NMS
                t = time_synchronized()
                if self.config.arch != "yolov8":
                    output = non_max_suppression(
                        inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=False
                    )

                elif self.config.arch == "yolov8":
                    output = yolov8ops.non_max_suppression(
                        inf_out,
                        conf_thres,
                        iou_thres,
                        labels=[],
                        multi_label=True,
                        agnostic=self.config.single_cls,
                        max_det=300,
                    )

                t1 += time_synchronized() - t

            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if pred is None:
                    if nl:
                        stats.append(
                            (
                                torch.zeros(0, niou, dtype=torch.bool),
                                torch.Tensor(),
                                torch.Tensor(),
                                tcls,
                            )
                        )
                    continue

                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = torch.zeros(
                    pred.shape[0], niou, dtype=torch.bool, device=device
                )
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (
                            (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        )  # prediction indices
                        pi = (
                            (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                        )  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(
                                1
                            )  # best ious, indices

                            # Append detections
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d not in detected:
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if (
                                        len(detected) == nl
                                    ):  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            p, r, ap50, ap = (
                p[:, 0],
                r[:, 0],
                ap[:, 0],
                ap.mean(1),
            )  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(
                stats[3].astype(np.int64), minlength=nc
            )  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = "%20s" + "%12.3g" * 6  # print format
        print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if verbose and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (
            imgsz,
            imgsz,
            batch_size,
        )  # tuple
        if not self.config.mode == "train":
            print(
                "Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g"
                % t
            )

        # Return results
        model.float()  # for training
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        del model
        torch.cuda.empty_cache()
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
