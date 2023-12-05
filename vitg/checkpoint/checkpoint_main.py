import os
from vitg.network.backbone.vitgyolov7.utils.torch_utils import (
    intersect_dicts as intersect_dictsyolov7,
)
import torch.nn as nn
import torch


class Checkpoint:
    def __init__(self, config) -> None:
        self.config = config

    def load_checkpoint(self, last_checkpoint: str, device):
        """It returns the checkpoint file to load and a boolean to indicate if it is a resume train or not.

        Args:
            last_checkpoint (_type_):  Chekcpoint location

        Returns:
            _type_: _description_
        """
        self.device = device
        if len(self.config.checkpoint_file) != 0:
            resume_train = True
            weights = os.path.join(last_checkpoint, self.config.checkpoint_file)
            if os.path.exists(weights):
                resume_train = True
            else:
                resume_train = False
                weights = ""
        elif self.config.load_checkpoint:
            resume_train = True
            weights = os.path.join(last_checkpoint, "checkpoint.pt")
            if os.path.exists(weights):
                resume_train = True
            else:
                resume_train = False
                weights = ""
        else:
            resume_train = False
            weights = ""
        return resume_train, weights

    def save_checkpoint(self, last_checkpoint, ckpt, epoch):
        if epoch % self.config.checkpoint_step == 0:
            torch.save(ckpt, os.path.join(last_checkpoint, "checkpoint.pt"))

    def make_checkpoint_dict_save(
        self,
        model,
        optimizer,
        epoch,
        best_train_map,
        best_val_map,
        end_time,
        final_epoch,
    ):
        ckpt = {
            "epoch": epoch,
            "best_train_map": best_train_map,
            "best_val_map": best_val_map,
            "model": model.state_dict(),
            "optimizer": None if final_epoch else optimizer.state_dict(),
            "end_time": end_time,
        }
        self.save_checkpoint(ckpt, epoch)
        return ckpt

    def check_retrain(
        self,
        epochs,
        weights,
        exclude,
        resume_train,
        model,
        optimizer,
    ):
        start_epoch, best_train_map, best_val_map = 0, 0.0, 0.0

        if weights is not "":
            print(f"load with pretrain weight: {weights}")
            ckpt = torch.load(weights, map_location=self.device)  # load checkpoint
            state_dict = intersect_dictsyolov7(
                ckpt["model"], model.state_dict(), exclude=exclude
            )  # intersect
            model.load_state_dict(state_dict, strict=False)
            print(
                "Transferred %g/%g items from %s"
                % (len(state_dict), len(model.state_dict()), weights)
            )  # report
            if not resume_train:
                # resume training need to load ckpt later
                del ckpt
            del state_dict
        else:
            print("train from scratch")
            # weight_initialization
            for name, m in model.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    if self.config.weight_initialization == "kaiming":
                        nn.init.kaiming_normal_(m.weight)
                    elif self.config.weight_initialization == "xavier":
                        nn.init.xavier_uniform_(m.weight)
            resume_train = False

        if resume_train:
            # Optimizer
            if ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
                best_train_map = ckpt["best_train_map"]
                best_val_map = ckpt["best_val_map"]

            # Epochs
            start_epoch = ckpt["epoch"]
            # start_epoch = ckpt['epoch']+1
            if epochs < start_epoch:
                print(
                    "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                    % (weights, ckpt["epoch"], epochs)
                )
                epochs += ckpt["epoch"]  # finetune additional epochs
            del ckpt

        return start_epoch, best_train_map, best_val_map
