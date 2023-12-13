import os
import shutil
from pathlib import Path

import numpy as np
import ruamel.yaml
import torch
import yaml

from vitg.network.backbone.vitgyolov8.yolo.utils import ops as yolov8ops

# build model currently support yolov4csp
from vitg.symbols.network import Darknet

# dataloader
from vitg.utils.datasets import create_test_dataloader

# loss function for yolov4csp

yamldy = ruamel.yaml.YAML()
import datetime
import math

# import vitgyolor.test as testyolov7
from vitg.network.backbone.vitgyolor.models.yolo import Model as Modelyolor
from vitg.network.backbone.vitgyolov7.models.yolo import Model as Modelyolov7

# ComputeLoss
from vitg.network.backbone.vitgyolov8.nn.tasks import DetectionModel as yolov8puremodel

# yolov8
from vitg.network.backbone.vitgyolov8.yolo.utils import yaml_load
from vitg.utils.general import (
    check_img_size,
    non_max_suppression_local,
    output_to_target_local_convert,
)
from vitg.utils.torch_utils import select_device
import logging


class Output:
    def __init__(self, config) -> None:
        self.save_config_yaml(config)

        for directory_list in ["log", "result", "model", "checkpoint"]:
            select_path = os.path.join(os.path.dirname(config.out_dir), directory_list)
            if not os.path.exists(select_path):
                os.makedirs(select_path)

        if not os.path.exists(os.path.dirname(config.log_path)):
            os.makedirs(os.path.dirname(config.log_path))

        config.logger = logging.getLogger("my_logger")
        config.forceclose = 0
        logging.basicConfig(level=logging.DEBUG)
        fh = logging.FileHandler(config.log_path, "a")

        config.logger.addHandler(fh)

    def save_training_log(
        self,
        config,
        out_dir,
        counterj,
        currentepoch,
        elapsed_time,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    ):
        # Save to YAML File

        if math.isnan(train_loss):
            train_loss = 0
        if math.isnan(train_acc):
            train_acc = 0
        if math.isnan(val_loss):
            val_loss = 0
        if math.isnan(val_acc):
            val_acc = 0

        # log_yaml_file = os.path.join(out_dir,'/log/train_log.yaml')
        log_yaml_file = os.path.join(out_dir, "log", f"{config.name}train_log.yaml")
        print("out_dir")
        print(out_dir)
        print(log_yaml_file)

        if os.path.exists(log_yaml_file):
            with open(log_yaml_file, "r") as file:
                docori = yaml.load(file, Loader=yaml.FullLoader)
            # time_obj = time.strptime(timestart, '%H:%M:%S')
            elapsed_time_ori = docori["log"][-1]["Elapsed"]

            if int(docori["log"][-1]["Epoch"]) >= int(currentepoch):
                return None

            # fix bug of repeat save same epoch after task resume

            tori = datetime.datetime.strptime(elapsed_time_ori, "%H:%M:%S")
            # ...and use datetime's hour, min and sec properties to build a timedelta
            elapsed_time_ori = datetime.timedelta(
                hours=tori.hour, minutes=tori.minute, seconds=tori.second
            )

            t = datetime.datetime.strptime(elapsed_time, "%H:%M:%S")
            # ...and use datetime's hour, min and sec properties to build a timedelta
            elapsed_time = datetime.timedelta(
                hours=t.hour, minutes=t.minute, seconds=t.second
            )
            elapsed_time3 = elapsed_time_ori + elapsed_time
            total_seconds = elapsed_time3.total_seconds()

            elapsed_time = str(datetime.timedelta(seconds=total_seconds))[-8:]
            # save train logger
            config.logger.info(
                f"epoch {currentepoch} - Elapsed, TrainLoss, TrainMap: {elapsed_time}, {train_loss}, {train_acc}"
            )

        elif counterj == 0:
            doc = {"log": []}
            with open(log_yaml_file, "w") as file:
                yaml.dump(doc, file)
        save_temp = {
            "Epoch": currentepoch,
            "Elapsed": elapsed_time,
            "Training Loss": train_loss,
            "Training Accuracy": train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
        }

        with open(log_yaml_file, "r") as file:
            # doc = yaml.safe_load(file)
            doc = yaml.load(file, Loader=yaml.FullLoader)
            doc["log"].append(save_temp)

        with open(log_yaml_file, "w") as file:
            yaml.dump(doc, file)

    def save_summary(self, config, run, best_train_acc, best_val_acc):
        # Generate summary
        summary = {
            "Number of epoch": [],
            "Best train accuracy": [],
            "Best validation accuracy": [],
        }
        if config.mode == "train":
            summary["Number of epoch"] = run
            summary["Best train accuracy"] = round(best_train_acc, 4)
            summary["Best validation accuracy"] = round(best_val_acc, 4)
        if config.mode == "val":
            summary["Best validation accuracy"] = round(best_val_acc, 4)

        if not os.path.exists(os.path.dirname(config.out_dir)):
            os.makedirs(os.path.dirname(config.out_dir))

        # summary_file = config.out_dir + '/result/' + config.mode + '_summary.yaml'
        summary_file = os.path.join(
            config.out_dir, "result", f"{config.mode}_summary.yaml"
        )

        if os.path.exists(summary_file):
            os.remove(summary_file)

        with open(summary_file, "w") as outfile:
            yaml.dump(summary, outfile)

        summary_filev2 = os.path.join(config.out_dir, "result", "summary.yaml")

        train_yamlv2_out = {
            "info": {
                "id": "111ageval",
                "type": "train",
                "description": "",
                "created": "",
                "dataset": {"id": "111ageval", "name": "", "version": ""},
            },
            "summary": [
                {"metric": "epoch", "value": run},
                {"metric": "map", "value": round(best_train_acc, 4)},
            ],
        }

        val_yamlv2_out = {
            "info": {
                "id": "111ageval",
                "type": "val",
                "description": "",
                "created": "",
                "dataset": {"id": "111ageval", "name": "", "version": ""},
            },
            "summary": [
                {"metric": "epoch", "value": run},
                {"metric": "map", "value": round(best_val_acc, 4)},
            ],
        }

        with open(summary_filev2, "w") as f_val_yaml:
            yamldy.dump(train_yamlv2_out, f_val_yaml)
            f_val_yaml.write("---" + "\n")
            yamldy.dump(val_yamlv2_out, f_val_yaml)

    def save_config_yaml(self, config):
        model_type_dict = {
            "yolov4csp": "YOLO",
            "yolov7": "YOLO",
            "yolor": "YOLO",
            "mobilenetssd": "mobilenetssd",
            "yolov8": "yolov8",
        }

        output = {
            "framework_type": ["Pytorch"],
            "input_type": ["Single Image"],
            "model_type": [model_type_dict.get(config.arch)],
            "img_size": config.img_size,
            "transforms": {},
            "output_type": ["Detection"],
            "category_list": [config.category_dic],
        }

        output_config = os.path.join(config.model_dir, f"{config.name}config.yaml")
        if not os.path.exists(os.path.dirname(output_config)):
            os.makedirs(os.path.dirname(output_config))
        with open(output_config, "w") as outfile:
            yaml.dump(output, outfile)

    def save_output_yaml(self):
        if self.mode == "train":
            train_out_file = os.path.join(self.out_dir, "result", "train_out.yaml")

            self.logger.info("start to generate train_out details")
            self.logger.info(train_out_file)
            if os.path.exists(train_out_file):
                os.remove(train_out_file)

            with open(train_out_file, "w") as f_yaml:
                yamldy.dump(self.train_out, f_yaml)

            val_out_file = os.path.join(self.out_dir, "result", "val_out.yaml")

            self.logger.info("start to generate val_out details")
            self.logger.info(val_out_file)
            if os.path.exists(val_out_file):
                os.remove(val_out_file)

            with open(val_out_file, "w") as f_val_yaml:
                yamldy.dump(self.val_out, f_val_yaml)

        elif self.mode == "val":
            val_out_file = os.path.join(self.out_dir, "result", f"{self.mode}_out.yaml")

            self.logger.info(val_out_file)
            self.logger.info("start to generate")
            self.logger.info(self.mode)
            if os.path.exists(val_out_file):
                os.remove(val_out_file)

            with open(val_out_file, "w") as f_val_yaml:
                yamldy.dump(self.val_out, f_val_yaml)

        elif self.mode == "test":
            test_out_file = os.path.join(
                self.out_dir, "result", f"{self.mode}_out.yaml"
            )

            self.logger.info(test_out_file)
            self.logger.info("start to generate")
            self.logger.info(self.mode)
            if os.path.exists(test_out_file):
                os.remove(test_out_file)

            with open(test_out_file, "w") as f_test_yaml:
                yamldy.dump(self.test_out, f_test_yaml)

    def generate_train_out_yaml(
        self,
        weights=None,
        batch_size=16,
        imgsz=640,
        conf_thres=0.05,
        iou_thres=0.6,
        save_json=False,
        single_cls=False,
        augment=False,
        verbose=False,
        model=None,
        dataloader=None,
        save_dir="",
        class_num=1,
        merge=False,
        save_txt=False,
        config=None,
    ):
        # Initialize/load model and set device
        training = model is not None
        # initial val_out.yaml

        yaml_out_path = os.path.join(
            config.out_dir, "result", f"{config.mode}_out.yaml"
        )
        config.logger.info(yaml_out_path)
        config.logger.info("start to generate")
        config.logger.info(config.mode)
        if os.path.exists(yaml_out_path):
            os.remove(yaml_out_path)

        yaml_out_temp1 = {"info": {"dataset_id": "111ageval"}, "output_images": []}

        if training:  # called by train.py
            device = next(model.parameters()).device  # get model device

        # Half
        half = device.type != "cpu"  # half precision only supported on CUDA
        if half:
            model.half()

        # Configure
        model.eval()

        nc = class_num
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        seen = 0
        names = [str(item) for item in range(nc)]
        # coco91class = coco80_to_coco91_class()
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
        jdict, stats, ap, ap_class = [], [], [], []
        # print("img_size")
        # print(imgsz)
        # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_valoutyml, desc=s)):

        dump_yaml_temp = []
        with open(yaml_out_path, "w+") as f:
            f.write(yaml.dump(yaml_out_temp1))
            f.write("output:" + "\n")

            for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
                new_image_width = imgsz
                new_image_height = imgsz
                old_image_height = img.shape[-2]
                old_image_width = img.shape[-1]
                x_center = (imgsz - img.shape[-1]) // 2
                y_center = (imgsz - img.shape[-2]) // 2

                img_repalce = torch.full(
                    (img.shape[0], img.shape[1], imgsz, imgsz), 114
                )

                img_repalce[
                    :,
                    :,
                    y_center : y_center + old_image_height,
                    x_center : x_center + old_image_width,
                ] = img
                img = img_repalce

                # print("img type")
                # print(type(img.shape))

                img = img.to(device, non_blocking=True)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)
                nb, _, height, width = img.shape  # batch size, channels, height, width
                whwh = torch.Tensor([width, height, width, height]).to(device)

                for batch_img_idx in range(img.shape[0]):
                    w0 = shapes[batch_img_idx][0][1]
                    ori_w = shapes[batch_img_idx][0][1]
                    h0 = shapes[batch_img_idx][0][0]
                    ori_h = shapes[batch_img_idx][0][0]

                    img_mini = img[batch_img_idx].contiguous().view(1, 3, imgsz, imgsz)
                    with torch.no_grad():
                        # print("img size")
                        # print(img.size())
                        if config.arch != "yolov8":
                            # inf_out, train_out = model(img, augment=augment)
                            inf_out, train_out = model(img_mini, augment=False)
                            output = non_max_suppression_local(
                                inf_out,
                                conf_thres=conf_thres,
                                iou_thres=iou_thres,
                                merge=False,
                            )
                        else:
                            inf_out = model(img_mini)
                            output = yolov8ops.non_max_suppression(
                                inf_out,
                                conf_thres,
                                iou_thres,
                                labels=[],
                                multi_label=True,
                                agnostic=config.single_cls,
                                max_det=300,
                            )

                    sample_data = output_to_target_local_convert(
                        output, old_image_width, old_image_height
                    )

                    sample_data_1 = [
                        item for item in sample_data if item[6] > conf_thres
                    ]  # take score above theshold

                    sample_data_0_newplot = []
                    sample_data_0_newplot_abs = []
                    # for boxs in sample_data_1:
                    for annidx, boxs in enumerate(sample_data_1):
                        conf = str(boxs[6])
                        gt = np.int16(boxs[1])
                        temp_a = {
                            "input_id": str(paths[batch_img_idx]),
                            "bbox": [
                                float((boxs[2] - boxs[4] / 2)),
                                float((boxs[3] - boxs[5] / 2)),
                                float(boxs[4]),
                                float(boxs[5]),
                            ],
                            "segmentation": [],
                            "points": [],
                            "category_id": [config.category_dic[int(gt)]],
                            "category_confidence_value": [float(conf)],
                            "label": [],
                            "label_conf_value": [],
                            "output_image_id": "",
                            "output_id": f"1{str(annidx)}{str(paths[batch_img_idx])}",
                        }
                        dump_yaml_temp.append(temp_a)

                        # new_line=[]
                        xmin = boxs[2] - boxs[4] / 2
                        xmin = xmin.astype(float).item()
                        xminabs = xmin * w0

                        ymin = boxs[3] - boxs[5] / 2
                        ymin = ymin.astype(float).item()
                        yminabs = ymin * h0

                        xmax = boxs[2] + boxs[4] / 2
                        xmax = xmax.astype(float).item()
                        ymax = boxs[3] + boxs[5] / 2
                        ymax = ymax.astype(float).item()
                        wabs = (xmax - xmin) * w0
                        habs = (ymax - ymin) * h0

                        conf = boxs[6].astype(float).item()
                        gt = np.int16(boxs[1])
                        new_line = [xmin, ymin, xmax, ymax, conf, gt]
                        new_line_abs = [xminabs, yminabs, wabs, habs, conf, gt]
                        # take detected body in [xmin, ymin, w, h, conf, class/label] format
                        sample_data_0_newplot.append(new_line)
                        sample_data_0_newplot_abs.append(new_line_abs)

                    if batch_i % batch_size == 0 or batch_i == len(dataloader) - 1:
                        if len(dump_yaml_temp) != 0:
                            # print(batch_i)
                            f.write(yaml.dump(dump_yaml_temp))
                        dump_yaml_temp = []
        del model
        torch.cuda.empty_cache()
        return None

    def generate_val_out_yaml(
        self,
        dataset,
        weights=None,
        batch_size=16,
        imgsz=640,
        conf_thres=0.05,
        iou_thres=0.6,
        save_json=False,
        single_cls=False,
        augment=False,
        verbose=False,
        model=None,
        dataloader=None,
        save_dir="",
        class_num=1,
        merge=False,
        save_txt=False,
        config=None,
    ):
        # Initialize/load model and set device
        # training = model is not None
        training = config.mode == "train"

        if config.mode == "train":
            yaml_out_path = os.path.join(config.out_dir, "result", "val_out.yaml")
        else:
            weights = config.pretrained_model
            yaml_out_path = os.path.join(
                config.out_dir, "result", f"{config.mode}_out.yaml"
            )
        config.logger.info(yaml_out_path)
        config.logger.info("start to generate")
        config.logger.info(config.mode)
        if os.path.exists(yaml_out_path):
            os.remove(yaml_out_path)

        yaml_out_temp1 = {"info": {"dataset_id": "111ageval"}, "output_images": []}

        if training:  # called by train.py
            device = next(model.parameters()).device  # get model device
        else:  # called directly
            device = select_device(config.device, batch_size=1)
            # merge, save_txt = config.merge, config.save_txt  # use Merge NMS, save *.txt labels
            save_txt = config.save_txt  # use Merge NMS, save *.txt labels
            if save_txt:
                out = Path("inference/output")
                if os.path.exists(out):
                    shutil.rmtree(out)  # delete output folder
                os.makedirs(out)  # make new output folder

            if config.arch == "yolov4csp":
                # load model under yolov4csp
                model = Darknet(config.cfg, class_num).to(device)
            elif config.arch == "yolov7":
                # load model under yolov7
                model = Modelyolov7(config.cfg, ch=3, nc=class_num, anchors=None).to(
                    device
                )  # create
            elif config.arch == "yolor":
                # load model under yolov7
                model = Modelyolor(config.cfg, ch=3, nc=config.class_number).to(
                    device
                )  # create
            elif config.arch == "yolov8":
                # cfg2 = check_yaml(config.cfg)  # check YAML
                cfg_dict = yaml_load(config.cfg, append_filename=True)  # model dict
                cfg_dict["nc"] = config.class_number
                model = yolov8puremodel(cfg_dict).to(device)

                # load model
            print("load weights from from path1")
            print(weights)
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            ckpt["model"] = {
                k: v
                for k, v in ckpt["model"].items()
                if model.state_dict()[k].numel() == v.numel()
            }
            model.load_state_dict(ckpt["model"], strict=False)

            imgsz = check_img_size(imgsz[0], s=32)  # check img_size

        # Half
        half = device.type != "cpu"  # half precision only supported on CUDA
        if half:
            model.half()

        # Configure
        model.eval()

        nc = class_num
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        # Dataloader
        if not training:
            # print("use new dataloader")
            stride_test = 1
            padding_test = 0.0
            img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            _ = (
                model(img.half() if half else img) if device.type != "cpu" else None
            )  # run once
            dataloader = create_test_dataloader(
                dataset,
                imgsz,
                1,
                stride_test,
                config,
                hyp=None,
                augment=False,
                cache=False,
                pad=padding_test,
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
        jdict, stats, ap, ap_class = [], [], [], []

        dump_yaml_temp = []
        with open(yaml_out_path, "w+") as f:
            f.write(yaml.dump(yaml_out_temp1))
            f.write("output:" + "\n")

            for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
                new_image_width = imgsz
                new_image_height = imgsz
                old_image_height = img.shape[-2]
                old_image_width = img.shape[-1]
                x_center = (imgsz - img.shape[-1]) // 2
                y_center = (imgsz - img.shape[-2]) // 2

                img_repalce = torch.full(
                    (img.shape[0], img.shape[1], imgsz, imgsz), 114
                )

                img_repalce[
                    :,
                    :,
                    y_center : y_center + old_image_height,
                    x_center : x_center + old_image_width,
                ] = img
                img = img_repalce

                img = img.to(device, non_blocking=True)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)
                nb, _, height, width = img.shape  # batch size, channels, height, width
                whwh = torch.Tensor([width, height, width, height]).to(device)

                for batch_img_idx in range(img.shape[0]):
                    w0 = shapes[batch_img_idx][0][1]
                    ori_w = shapes[batch_img_idx][0][1]
                    h0 = shapes[batch_img_idx][0][0]
                    ori_h = shapes[batch_img_idx][0][0]
                    img_mini = img[batch_img_idx].contiguous().view(1, 3, imgsz, imgsz)

                    with torch.no_grad():
                        if config.arch != "yolov8":
                            inf_out, train_out = model(img_mini, augment=False)
                            output = non_max_suppression_local(
                                inf_out,
                                conf_thres=conf_thres,
                                iou_thres=iou_thres,
                                merge=False,
                            )
                        else:
                            inf_out = model(img_mini)
                            output = yolov8ops.non_max_suppression(
                                inf_out,
                                conf_thres,
                                iou_thres,
                                labels=[],
                                multi_label=True,
                                agnostic=config.single_cls,
                                max_det=300,
                            )

                    sample_data = output_to_target_local_convert(
                        output, old_image_width, old_image_height
                    )
                    sample_data_1 = [
                        item for item in sample_data if item[6] > conf_thres
                    ]  # take score above theshold

                    sample_data_0_newplot = []
                    sample_data_0_newplot_abs = []
                    # for boxs in sample_data_1:
                    for annidx, boxs in enumerate(sample_data_1):
                        conf = str(boxs[6])
                        # conf = np.int(boxs[6])
                        gt = np.int16(boxs[1])
                        temp_a = {
                            "input_id": str(paths[batch_img_idx]),
                            "bbox": [
                                float((boxs[2] - boxs[4] / 2)),
                                float((boxs[3] - boxs[5] / 2)),
                                float(boxs[4]),
                                float(boxs[5]),
                            ],
                            "segmentation": [],
                            "points": [],
                            "category_id": [config.category_dic[int(gt)]],
                            "category_confidence_value": [float(conf)],
                            "label": [],
                            "label_conf_value": [],
                            "output_image_id": "",
                            "output_id": f"1{str(annidx)}{str(paths[batch_img_idx])}",
                        }
                        dump_yaml_temp.append(temp_a)

                        xmin = boxs[2] - boxs[4] / 2
                        xmin = xmin.astype(float).item()
                        xminabs = xmin * w0

                        ymin = boxs[3] - boxs[5] / 2
                        ymin = ymin.astype(float).item()
                        yminabs = ymin * h0

                        xmax = boxs[2] + boxs[4] / 2
                        xmax = xmax.astype(float).item()
                        ymax = boxs[3] + boxs[5] / 2
                        ymax = ymax.astype(float).item()
                        wabs = (xmax - xmin) * w0
                        habs = (ymax - ymin) * h0

                        conf = boxs[6].astype(float).item()
                        gt = np.int16(boxs[1])
                        new_line = [xmin, ymin, xmax, ymax, conf, gt]
                        new_line_abs = [xminabs, yminabs, wabs, habs, conf, gt]
                        # take detected body in [xmin, ymin, w, h, conf, class/label] format
                        sample_data_0_newplot.append(new_line)
                        sample_data_0_newplot_abs.append(new_line_abs)

                    if batch_i % batch_size == 0 or batch_i == len(dataloader) - 1:
                        if len(dump_yaml_temp) != 0:
                            # print(batch_i)
                            f.write(yaml.dump(dump_yaml_temp))
                        dump_yaml_temp = []
        del model
        torch.cuda.empty_cache()
        return None
