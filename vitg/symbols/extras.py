import signal
from shutil import copyfile as copyfilevade

import numpy as np
import ruamel.yaml
from vitg.utils.layers import *
from vitg.symbols.yolo_pth import Darknet as Darknet2

yamldy = ruamel.yaml.YAML()

# for python3:
try:
    range
except NameError:
    range = range


from torchvision.ops.boxes import box_convert, box_iou

from vitg.network.backbone.mobilenetssdv3.vision.ssd.mobilenet_v3_ssd_lite import (
    create_mobilenetv3_ssd_lite,
)

# ComputeLoss
from vitg.network.backbone.vitgyolor.models.yolo import Model as Modelyolor
from vitg.network.backbone.vitgyolov7.models.yolo import Model as Modelyolov7
from vitg.network.backbone.vitgyolov8.nn.tasks import DetectionModel as yolov8puremodel

# yolov8
from vitg.network.backbone.vitgyolov8.yolo.utils import yaml_load
from vitg.network.backbone.vitgyolov8.yolo.utils.loss import BboxLoss
from vitg.network.backbone.vitgyolov8.yolo.utils.ops import xywh2xyxy as xywh2xyxyv8
from vitg.network.backbone.vitgyolov8.yolo.utils.tal import (
    TaskAlignedAssigner,
    dist2bbox,
    make_anchors,
)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience: int = 7, verbose: bool = False, delta: float = 0
    ) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0


class Force_close:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self) -> None:
        self.forceclose = 0
        signal.signal(signal.SIGINT, self.handler)
        signal.signal(signal.SIGHUP, self.handler)
        signal.signal(signal.SIGTERM, self.handler)

    def handler(self, signum, frame=None) -> None:
        self.forceclose = 1
        # self.signum = signum
        print("Signal handler called with signal", signum)
        try:
            self.logger.info(f"Force Close Detected => {signum}")
        except Exception:
            print("No logger set")
        print("Force Close Detected")

    def check_close(self) -> int:
        return self.forceclose

    def reset_forceclose(self):
        self.forceclose = 0


class Lossv8func:
    def __init__(self, model, classuse):  # model must be de-paralleled
        device = next(model.parameters()).device  # get model device
        # h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.hyp = h
        self.stride = m.stride  # model strides
        # print("stride use")
        # print(self.stride)
        self.nc = classuse  # number of classes
        # print("class numer")
        # print(self.nc)
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        # roll_out_thr = h.min_memory if h.min_memory > 1 else 64 if h.min_memory else 0  # 64 is default
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0, roll_out_thr=roll_out_thr
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxyv8(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, targets):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        box_gain = 7.5  # box loss gain
        cls_gain = 0.5  # cls loss gain (scale with pixels)
        dfl_gain = 1.5  # dfl loss gain
        loss[0] *= box_gain  # box gain
        loss[1] *= cls_gain  # cls gain
        loss[2] *= dfl_gain  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


def preprocess_v8(targets, batch_size, scale_tensor):
    if targets.shape[0] == 0:
        out = torch.zeros(batch_size, 0, 5)
    else:
        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5)
        for j in range(batch_size):
            matches = i == j
            if n := matches.sum():
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxyv8(out[..., 1:5].mul_(scale_tensor))
    return out


def bbox_decode(anchor_points, pred_dist, proj):
    # if self.use_dfl:
    b, a, c = pred_dist.shape  # batch, anchors, channels
    pred_dist = (
        pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
    )
    # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
    # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
    return dist2bbox(pred_dist, anchor_points, xywh=False)


def save_c_model_val2(opt, mode_path):
    """
    this use pytorch python api(libtorch) to convert the python model to c model
    see example:https://pytorch.org/tutorials/advanced/cpp_export.html
    this is a new function for pytorch 1.1.0
    """

    if opt.arch == "yolov4csp":
        # load model under yolov4csp
        modeltest = Darknet2(opt.cfg, opt.class_number)
    elif opt.arch == "yolov7":
        # load model under yolov7
        modeltest = Modelyolov7(
            opt.cfg, ch=3, nc=opt.class_number, anchors=None
        )  # create

    elif opt.arch == "yolor":
        # load model under yolov7
        modeltest = Modelyolor(opt.cfg, ch=3, nc=opt.class_number)  # create
    elif opt.arch == "mobilenetssd":
        # print("use ssd in mobilenet")
        # modeltest = mobilessdv2(backbone=MobileNetV2(), num_classes=opt.class_number+1)
        modeltest = create_mobilenetv3_ssd_lite(opt.class_number + 1)
        # modeltest = mobilessdv2(backbone=MobileNetV2(), num_classes=opt.class_number+1)

        # mobilessdv3
        # modeltest = mobilessdv2(backbone=MobileNetV2(), num_classes=opt.class_number+1)
        # return None
    elif opt.arch == "yolov8":
        # cfgv8 = check_yaml(opt.cfg)  # check YAML
        cfg_dictv8 = yaml_load(opt.cfg, append_filename=True)  # model dict
        cfg_dictv8["nc"] = opt.class_number
        modeltest = yolov8puremodel(cfg_dictv8)
        # del cfgv8
        del cfg_dictv8

        # load model
    # print("load weights from from path1")
    # print(weights)
    ckpttest = torch.load(mode_path)
    modeltest.load_state_dict(ckpttest["model"], strict=False)
    # modeltest.cuda()
    modeltest.eval()
    if opt.arch == "mobilenetssd":
        modeltest.cpu()
        input_shape = [1, 3, opt.img_size[0], opt.img_size[1]]
        # output_model_name_GPU = "c++_eyelevel_script.pt"
        example = (torch.rand(input_shape) * 255).cpu()
        # example = (torch.rand(input_shape) * 255)
        example = example.contiguous().view(1, 3, opt.img_size[0], opt.img_size[1])
        opt.logger.info(
            f"Prepare go generate model to C with dumy input shape{example.shape}"
        )

        outdummy = modeltest(example)
        traced_script_module = torch.jit.trace(modeltest, example, check_trace=False)
        # output_model_name_GPU = os.path.join(opt.model_dir, 'model_infer.pt')
        output_model_name_GPU = mode_path.replace(".pth", "_infer.pt")
        model_path_pth_inference = mode_path.replace(".pth", "_infer.pth")
        # torch.save(modeltest,model_path_pth_inference)
        copyfilevade(mode_path, model_path_pth_inference)
        print("save c model path")
        print(output_model_name_GPU)
        print(model_path_pth_inference)
        traced_script_module.save(output_model_name_GPU)

        opt.logger.info("Exported model to C")
        return None

    model_path_pth_inference = mode_path.replace(".pth", "_infer.pth")
    torch.save(modeltest, model_path_pth_inference)
    modelcpp = torch.load(model_path_pth_inference)
    # print("reload with again to solve batch issue")
    modelcpp.cuda()
    modelcpp.eval()

    input_shape = [1, 3, opt.img_size[0], opt.img_size[1]]
    # output_model_name_GPU = "c++_eyelevel_script.pt"
    example = (torch.rand(input_shape) * 255).cuda()
    # example = (torch.rand(input_shape) * 255)
    example = example.contiguous().view(1, 3, opt.img_size[0], opt.img_size[1])
    opt.logger.info(
        f"Prepare go generate model to C with dumy input shape{example.shape}"
    )

    outdummy = modelcpp(example)
    traced_script_module = torch.jit.trace(modelcpp, example, check_trace=False)
    # output_model_name_GPU = os.path.join(opt.model_dir, 'model_infer.pt')
    output_model_name_GPU = mode_path.replace(".pth", "_infer.pt")
    print("save c model path")
    print(output_model_name_GPU)
    traced_script_module.save(output_model_name_GPU)

    opt.logger.info("Exported model to C")

    del modeltest
    del ckpttest
    del model_path_pth_inference
    del modelcpp
    del traced_script_module
    return None


def encode_local(bboxes_in, labels_in, dboxes_default_encode, criteria=0.5):
    ious = box_iou(bboxes_in, dboxes_default_encode)
    best_dbox_ious, best_dbox_idx = ious.max(dim=0)
    best_bbox_ious, best_bbox_idx = ious.max(dim=1)

    # set best ious 2.0
    best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

    idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
    best_dbox_idx[best_bbox_idx[idx]] = idx

    # filter IoU > 0.5
    masks = best_dbox_ious > criteria
    labels_out = torch.zeros(3000, dtype=torch.long)
    labels_out[masks] = labels_in[best_dbox_idx[masks]]
    bboxes_out = dboxes_default_encode.clone()
    bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
    bboxes_out = box_convert(bboxes_out, in_fmt="xyxy", out_fmt="cxcywh")
    return bboxes_out, labels_out
