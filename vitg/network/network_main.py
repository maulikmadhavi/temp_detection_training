from vitg.network.backbone.mobilenetSSD.src.utils import generate_dboxes
from vitg.network.backbone.mobilenetssdv3.vision.ssd.mobilenet_v3_ssd_lite import (
    create_mobilenetv3_ssd_lite,
)
from vitg.network.backbone.vitgyolor.models.yolo import Model as Modelyolor
from vitg.network.backbone.vitgyolov7.models.yolo import Model as Modelyolov7
from vitg.network.backbone.vitgyolov7.utils.torch_utils import (
    intersect_dicts as intersect_dictsyolov7,
)
from vitg.network.backbone.vitgyolov8.nn.tasks import DetectionModel as Modelyolov8
from vitg.network.backbone.vitgyolov8.yolo.utils import yaml_load
from vitg.symbols.network import Darknet
from vitg.utils.constants import NBS

import ruamel.yaml
import torch
import torch.nn as nn

yamldy = ruamel.yaml.YAML()


class Network:
    def __init__(self, config, hyp, device, resume_train) -> None:
        self.config = config
        self.hyp = hyp
        self.device = device
        self.dboxes = []

        # Configure
        self.cuda = device.type != "cpu"
        self.nc = self.config.class_number
        self.names = [str(item) for item in range(self.nc)]

        # Model Initializer
        self.make_model()

        total_batch_size, weights = (
            self.config.total_batch_size,
            self.config.weights,
        )

        # ============ I do not understand what is nbs? why it is used? =======
        nbs = NBS  # nominal batch size
        self.accumulate = max(
            round(nbs / total_batch_size), 1
        )  # accumulate loss before optimizing
        if resume_train:
            self.hyp["weight_decay"] *= (
                total_batch_size * self.accumulate / nbs
            )  # scale weight_decay

        self.initializer(weights, resume_train)

    def do_forward(self, imgs: torch.Tensor, gloc=None, glabel=None) -> torch.Tensor:
        if self.config.arch == "mobilenetssd":
            plabel, ploc = self.model(imgs.to(torch.float))
            ploc, plabel = ploc.float(), plabel.float()
            gloc = gloc.transpose(1, 2).contiguous()
            ploc = ploc.view(self.config.batch_size, -1, 3000)
            plabel = plabel.view(self.config.batch_size, -1, 3000)
            return plabel, ploc, gloc, glabel
        else:
            return self.model(imgs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def make_model(self):
        if self.config.arch == "yolov4csp":
            self.model = Darknet(self.config.cfg, self.config.class_number).to(
                self.device
            )  # create
            self.exclude = []
        elif self.config.arch == "yolov7":
            self.model = Modelyolov7(
                self.config.cfg,
                ch=3,
                nc=self.config.class_number,
                anchors=self.hyp.get("anchors"),
            ).to(
                self.device
            )  # create
            self.exclude = ["anchor"]
        elif self.config.arch == "yolor":
            self.model = Modelyolor(
                self.config.cfg, ch=3, nc=self.config.class_number
            ).to(
                self.device
            )  # create
            self.exclude = ["anchor"]
        elif self.config.arch == "mobilenetssd":
            self.dboxes = generate_dboxes(model="ssdlite")
            self.exclude = []
            self.model = create_mobilenetv3_ssd_lite(self.config.class_number + 1).to(
                self.device
            )
        elif self.config.arch == "yolov8":
            cfg_dict = yaml_load(self.config.cfg, append_filename=True)  # model dict
            cfg_dict["nc"] = self.config.class_number
            self.model = Modelyolov8(cfg_dict).to(self.device)
            self.exclude = []

        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        # self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
        self.model.names = self.names

    def initializer(self, weights_path: str, resume_train: bool = False):
        if weights_path != "":
            print("load with pretrain weight:", weights_path)
            ckpt = torch.load(weights_path, map_location=self.device)  # load checkpoint
            state_dict = intersect_dictsyolov7(
                ckpt["model"], self.model.state_dict(), exclude=self.exclude
            )  # intersect
            # state_dict = {k: v for k, v in ckpt['model'].items() if self.model.state_dict()[k].numel() == v.numel()}
            self.model.load_state_dict(state_dict, strict=False)
            print(
                "Transferred %g/%g items from %s"
                % (len(state_dict), len(self.model.state_dict()), weights_path)
            )  # report
            if not resume_train:
                # resume training need to load ckpt later
                del ckpt
            del state_dict
        else:
            print("train from scratch")
            # weight_initialization
            for name, m in self.model.named_modules():
                # print(m)
                if self.config.weight_initialization == "kaiming":
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight)
                elif self.config.weight_initialization == "xavier":
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)

        self.pg0, self.pg1, self.pg2 = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()

        if self.config.arch == "yolov7":
            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    self.pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d):
                    self.pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    self.pg1.append(v.weight)  # apply decay
                if hasattr(v, "im"):
                    if hasattr(v.im, "implicit"):
                        self.pg0.append(v.im.implicit)
                    else:
                        self.pg0.extend(iv.implicit for iv in v.im)
                if hasattr(v, "imc"):
                    if hasattr(v.imc, "implicit"):
                        self.pg0.append(v.imc.implicit)
                    else:
                        self.pg0.extend(iv.implicit for iv in v.imc)
                if hasattr(v, "imb"):
                    if hasattr(v.imb, "implicit"):
                        self.pg0.append(v.imb.implicit)
                    else:
                        self.pg0.extend(iv.implicit for iv in v.imb)
                if hasattr(v, "imo"):
                    if hasattr(v.imo, "implicit"):
                        self.pg0.append(v.imo.implicit)
                    else:
                        self.pg0.extend(iv.implicit for iv in v.imo)
                if hasattr(v, "ia"):
                    if hasattr(v.ia, "implicit"):
                        self.pg0.append(v.ia.implicit)
                    else:
                        self.pg0.extend(iv.implicit for iv in v.ia)
                if hasattr(v, "attn"):
                    if hasattr(v.attn, "logit_scale"):
                        self.pg0.append(v.attn.logit_scale)
                    if hasattr(v.attn, "q_bias"):
                        self.pg0.append(v.attn.q_bias)
                    if hasattr(v.attn, "v_bias"):
                        self.pg0.append(v.attn.v_bias)
                    if hasattr(v.attn, "relative_position_bias_table"):
                        self.pg0.append(v.attn.relative_position_bias_table)
                if hasattr(v, "rbr_dense"):
                    if hasattr(v.rbr_dense, "weight_rbr_origin"):
                        self.pg0.append(v.rbr_dense.weight_rbr_origin)
                    if hasattr(v.rbr_dense, "weight_rbr_avg_conv"):
                        self.pg0.append(v.rbr_dense.weight_rbr_avg_conv)
                    if hasattr(v.rbr_dense, "weight_rbr_pfir_conv"):
                        self.pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
                    if hasattr(v.rbr_dense, "weight_rbr_1x1_kxk_idconv1"):
                        self.pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
                    if hasattr(v.rbr_dense, "weight_rbr_1x1_kxk_conv2"):
                        self.pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
                    if hasattr(v.rbr_dense, "weight_rbr_gconv_dw"):
                        self.pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
                    if hasattr(v.rbr_dense, "weight_rbr_gconv_pw"):
                        self.pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
                    if hasattr(v.rbr_dense, "vector"):
                        self.pg0.append(v.rbr_dense.vector)
        elif self.config.arch == "yolov4csp":
            for k, v in dict(self.model.named_parameters()).items():
                if ".bias" in k:
                    self.pg2.append(v)  # biases
                elif "Conv2d.weight" in k:
                    self.pg1.append(v)  # apply weight_decay
                else:
                    self.pg0.append(v)  # all else
        elif self.config.arch == "yolor":
            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    self.pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d):
                    self.pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    self.pg1.append(v.weight)  # apply decay
                if hasattr(v, "im"):
                    self.pg0.extend(iv.implicit for iv in v.im)
                if hasattr(v, "ia"):
                    self.pg0.extend(iv.implicit for iv in v.ia)
                if hasattr(v, "id"):
                    self.pg0.extend(iv.implicit for iv in v.id)
                if hasattr(v, "iq"):
                    self.pg0.extend(iv.implicit for iv in v.iq)
                if hasattr(v, "ix"):
                    self.pg0.extend(iv.implicit for iv in v.ix)
                if hasattr(v, "ie"):
                    self.pg0.extend(iv.implicit for iv in v.ie)
                if hasattr(v, "ic"):
                    self.pg0.append(v.ic.implicit)
        else:
            # self.pg0=self.model.parameters()
            for v in self.model.modules():
                if hasattr(v, "bias") and isinstance(
                    v.bias, nn.Parameter
                ):  # bias (no decay)
                    self.pg0.append(v.bias)
                if isinstance(v, bn):  # weight (no decay)
                    self.pg2.append(v.weight)
                elif hasattr(v, "weight") and isinstance(
                    v.weight, nn.Parameter
                ):  # weight (with decay)
                    self.pg1.append(v.weight)
