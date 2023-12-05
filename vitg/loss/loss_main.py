# ComputeLoss
from vitg.network.backbone.vitgyolov7.utils.loss import ComputeLossAuxOTA
from vitg.network.backbone.vitgyolov7.utils.loss import ComputeLoss as ComputeLossv7

from vitg.network.backbone.vitgyolor.utils.loss import (
    compute_loss as compute_loss_yolor,
)
from vitg.network.backbone.vitgyolov8 import YOLO as yolov8orimodel
from vitg.network.backbone.vitgyolov8.yolo.utils import yaml_load
from vitg.symbols.extras import (
    Lossv8func,
)
from vitg.network.backbone.mobilenetSSD.src.utils import Encoder
from vitg.network.backbone.mobilenetSSD.src.loss import Loss as SSDLoss
from vitg.symbols.loss import compute_loss
from vitg.network.backbone.vitgyolov8.yolo.utils.torch_utils import de_parallel
import torch
import yaml


class Loss:
    def __init__(self, config, model, dboxes, device):
        self.encoder = None
        self.criterion = None
        if config.arch == "yolov7":
            self.compute_loss_ota = ComputeLossAuxOTA(model)  # init loss class
        elif config.arch == "mobilenetssd":
            self.encoder = Encoder(dboxes)
            self.criterion = SSDLoss(dboxes)
        elif config.arch == "yolov8":
            cfg_dictinterv8dumpy = yaml_load(
                config.cfg, append_filename=True
            )  # model dict
            cfg_dictinterv8dumpy["nc"] = config.class_number
            interyamlsave = config.cfg.replace(".yaml", "_interv8.yaml")

            # with open(interyamlsave, "w") as file:
            #     documents = yaml.dump(cfg_dictinterv8dumpy, file)
            model_v8_loss = yolov8orimodel(interyamlsave)
            model_v8_loss.model.to(device)
            self.compute_lossv8 = Lossv8func(
                de_parallel(model_v8_loss.model), config.class_number
            )
            del model_v8_loss
        self.config = config
        self.model = model
        self.device = device

    def __call__(self, pred, targets, imgs):
        if self.config.arch == "yolov7":
            loss, loss_items = self.compute_loss_ota(
                pred, targets.to(self.device), imgs
            )  # loss scaled by batch_size
        elif self.config.arch == "yolov4csp":
            loss, loss_items = compute_loss(
                pred, targets.to(self.device), self.model
            )  # loss scaled by batch_size
            # compute_loss_yolor
        elif self.config.arch == "yolor":
            loss, loss_items = compute_loss_yolor(
                pred, targets.to(self.device), self.model
            )  # scaled by batch_size
        elif self.config.arch == "mobilenetssd":
            plabel, ploc, gloc, glabel = pred
            loss = self.criterion(ploc.cpu(), plabel.cpu(), gloc.cpu(), glabel.cpu())
            loss_items = None
            # As we do not have isolated loss for each type, i.e.,
            # loc, conf, cls, total: we repeat the total loss 4 times
        elif self.config.arch == "yolov8":
            loss, loss_items = self.compute_lossv8(pred, targets)
            loss_sum_yolov8 = sum(loss_items).detach().cpu().numpy()
            loss_items = torch.cat(
                (
                    loss_items.detach().cpu(),
                    torch.tensor([float(loss_sum_yolov8)]),
                ),
                0,
            )
            loss_items = loss_items.to(self.device)
        return loss, loss_items


class LossTest:
    def __init__(self, config, model, dboxes, device):
        self.encoder = None
        self.criterion = None
        if config.arch == "yolov7" and config.mode == "train":
            self.compute_lossv7 = ComputeLossv7(model)
        elif config.arch == "yolov8":
            cfg_dictinterv8dumpy = yaml_load(
                config.cfg, append_filename=True
            )  # model dict
            cfg_dictinterv8dumpy["nc"] = config.class_number
            interyamlsave = config.cfg.replace(".yaml", "_interv8.yaml")

            with open(interyamlsave, "w") as file:
                yaml.dump(cfg_dictinterv8dumpy, file)
            model_v8_loss = yolov8orimodel(interyamlsave)
            model_v8_loss.model.to(device)

            self.compute_lossv8 = Lossv8func(
                de_parallel(model_v8_loss.model), config.class_number
            )
            del model_v8_loss
        self.config = config
        self.model = model
        self.device = device

    def __call__(self, pred, targets, imgs):
        loss = torch.zeros(3, device=self.device)
        if self.config.arch == "yolov7":
            loss += self.compute_lossv7([x.float() for x in pred], targets)[1][
                :3
            ]  # box, obj, cls
        elif self.config.arch == "yolor":
            loss += compute_loss_yolor([x.float() for x in pred], targets, self.model)[
                1
            ][
                :3
            ]  # GIoU, obj, cls
        elif self.config.arch == "yolov4csp":
            loss += compute_loss([x.float() for x in pred], targets, self.model)[1][
                :3
            ]  # GIoU, obj, cls

        elif self.config.arch == "yolov8":
            _, loss_items = self.compute_lossv8(pred[1], targets)

            loss += loss_items.detach().to(self.device)
        return loss
