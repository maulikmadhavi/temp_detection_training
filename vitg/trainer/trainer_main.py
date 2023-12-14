from typing import List, Tuple
import torch
import numpy as np
from vitg.utils.constants import NBS
from vitg.symbols.extras import (
    encode_local,
)
import os
from vitg.network.backbone.mobilenetSSD.src.process import (
    evaluate_lmdb_outyaml_mobilenet,
    evaluate_mobilenet,
)

from vitg.symbols.extras import (
    EarlyStopping,
    Force_close,
    Lossv8func,
    save_c_model_val2,
)


class BaseTrainer:
    def run_warmup(self, ni, nw, epoch) -> None:
        """
        This function is used to warm up the model before training. It adjusts the learning rate and momentum
        based on the interpolation of the current iteration number and the total number of warmup iterations.

        Parameters:
        ni (int): Current iteration number.
        nw (int): Total number of warmup iterations.
        epoch (int): Current epoch number.

        Returns:
        None
        """
        if ni <= nw:
            xi = [0, nw]  # x interp
            accumulate = max(
                1,
                np.interp(ni, xi, [1, NBS / self.config.total_batch_size]).round(),
            )
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    ni,
                    xi,
                    [0.1 if j == 2 else 0.0, x["initial_lr"] * self.lf(epoch)],
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(ni, xi, [0.9, self.hyp["momentum"]])

    def get_gloc_glabel(self, targets, net):
        """
        Get global boxes and global labels for the given targets and network.

        Args:
            targets (torch.Tensor): The targets tensor containing batch-wise bounding box and label information.
            net: The network object.

        Returns:
            torch.Tensor: The global boxes tensor.
            torch.Tensor: The global labels tensor.
        """
        if self.config.arch == "mobilenetssd":
            all_batch_boxes = []
            all_batch_labels = []
            for batch_idx in range(self.config.batch_size):
                temp_batch = targets[targets[:, 0] == batch_idx]
                temp_boxes = temp_batch[:, 2:]

                temp_labels = temp_batch[:, 1] + 1
                updated_boxes = [
                    [
                        item[0] - item[2] / 2,
                        item[1] - item[3] / 2,
                        item[0] + item[2] / 2,
                        item[1] + item[3] / 2,
                    ]
                    for item in temp_boxes
                ]
                updated_boxes_tensor = torch.as_tensor(updated_boxes)
                if updated_boxes_tensor.shape[0] == 0:
                    dummy_box = torch.zeros(1, 4)
                    updated_boxes_tensor = torch.cat((updated_boxes_tensor, dummy_box))
                    dummy_label = torch.tensor([0])
                    temp_labels = torch.cat((temp_labels, dummy_label))

                encoded_boxes, encoded_labels = encode_local(
                    bboxes_in=updated_boxes_tensor,
                    labels_in=temp_labels.to(torch.long),
                    dboxes_default_encode=net.dboxes(order="ltrb"),
                )

                all_batch_boxes.append(encoded_boxes.numpy())
                all_batch_labels.append(encoded_labels.numpy())

            global_boxes = torch.tensor(all_batch_boxes)
            global_labels = torch.tensor(all_batch_labels)
        else:
            global_boxes, global_labels = None, None

        return global_boxes, global_labels

    def run_forward(self, imgs, targets, net, gloc, glabel, loss_fun, rank):
        with self.amp_autocast():
            output = net.do_forward(imgs, gloc, glabel)

            # targets
            loss, loss_items = loss_fun(output, targets, imgs)

            if rank != -1 or self.config.use_ddp:
                loss *= (
                    self.config.world_size
                )  # gradient averaged between devices in DDP mode

        if self.config.arch == "mobilenetssd":
            loss = loss.to(self.device)

        return loss, loss_items

    def run_backward(self, loss, scaler):
        if self.config.use_amp:
            # Backward
            scaler.scale(loss).backward()
        elif self.config.arch == "mobilenetssd":
            loss.backward()
        else:
            with torch.autograd.detect_anomaly():
                loss.backward(retain_graph=True)

    def optimizer_step(self, scaler, model):
        if self.config.use_amp:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        # if ema is not None:
        #     ema.update(model)

    # def save_nonmobilenet_output_yaml(
    #     self,
    #     imgsz_test: int,
    #     model: Model,
    #     best_val_model: Model,
    #     testloader_val_yml: DataLoader,
    #     testloader_train_yml: DataLoader,
    # ):
    #     self.output.generate_val_out_yaml(
    #         self.config.val_dataset,
    #         batch_size=self.config.yaml_batch_size,
    #         imgsz=imgsz_test,
    #         save_json=False,
    #         model=best_val_model,
    #         weights=self.config.weights,
    #         single_cls=self.config.single_cls,
    #         dataloader=testloader_val_yml,
    #         class_num=self.config.class_number,
    #         config=self.config,
    #     )
    #     # generate train_out.yml at final epoch
    #     self.output.generate_train_out_yaml(
    #         batch_size=self.config.yaml_batch_size,
    #         imgsz=imgsz_test,
    #         save_json=False,
    #         # model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
    #         model=model,
    #         weights=self.config.weights,
    #         single_cls=self.config.single_cls,
    #         dataloader=testloader_train_yml,
    #         class_num=self.config.class_number,
    #         config=self.config,
    #     )

    # def save_best_model(
    #     self,
    #     best_map: float,
    #     val_map_log: List[float],
    #     epoch: int,
    #     final_epoch: int,
    #     best_val: float,
    # ) -> None:
    #     best_train_map, best_val_map = best_map
    #     best_val_model = self.net.model

    #     # If the model with the best validation mAP is the same as the current model,
    #     # then save the model and its training and validation mAPs.
    #     if best_val_map == val_map_log:
    #         self.logger.info("save best val epoch model")
    #         ckpt = {
    #             "epoch": epoch,
    #             "best_val_map": best_val_map,
    #             "best_train_map": best_train_map,
    #             "model": best_val_model.state_dict(),
    #             "self.optimizer": None if final_epoch else self.optimizer.state_dict(),
    #         }

    #         torch.save(ckpt, best_val)
    #         save_c_model_val2(self.config, best_val)
    #     return best_val_model

    # def save_output_yaml(
    #     self,
    #     imgsz_test,
    #     model,
    #     best_val_model,
    #     testloader_val_yml,
    #     testloader_train_yml,
    #     testloader_noaug,
    #     dataloader,
    #     epoch,
    # ):
    #     if self.config.arch != "mobilenetssd":
    #         # generate val_out.yml when val result improve
    #         self.save_nonmobilenet_output_yaml(
    #             imgsz_test,
    #             model,
    #             best_val_model,
    #             testloader_val_yml,
    #             testloader_train_yml,
    #         )
    #     # del ema
    #     else:
    #         self.save_mobilenet_output_yaml(
    #             testloader_noaug, dataloader, epoch, model, best_val_model
    #         )

    # def save_mobilenet_output_yaml(
    #     self, testloader_noaug, dataloader, epoch, model, best_val_model
    # ):
    #     print("start save out yaml")
    #     self.config.yaml_out_path = os.path.join(
    #         self.config.out_dir,
    #         "result",
    #         f"{self.config.name}val_out.yaml",
    #     )
    #     evaluate_lmdb_outyaml_mobilenet(
    #         self.config,
    #         best_val_model,
    #         testloader_noaug,
    #         epoch,
    #         None,
    #         self.loss_fun.encoder,
    #         0.5,
    #         dboxes_default=self.net.dboxes(order="ltrb"),
    #     )
    #     print("save val out yaml done")
    #     self.config.yaml_out_path = os.path.join(
    #         self.config.out_dir,
    #         "result",
    #         f"{self.config.name}train_out.yaml",
    #     )
    #     evaluate_lmdb_outyaml_mobilenet(
    #         self.config,
    #         model,
    #         dataloader,
    #         epoch,
    #         None,
    #         self.loss_fun.encoder,
    #         0.5,
    #         dboxes_default=self.net.dboxes(order="ltrb"),
    #     )
    #     print("save train out yaml done")
