import math

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn

from vitg.utils.torch_utils import is_parallel

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(
    linewidth=320, formatter={"float_kind": "{:11.5g}".format}
)  # format short g, %precision=5
matplotlib.rc("font", **{"size": 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = (
        torch.zeros(1, device=device),
        torch.zeros(1, device=device),
        torch.zeros(1, device=device),
    )
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h["cls_pw"]])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h["obj_pw"]])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h["fl_gamma"]  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        if n := b.shape[0]:
            nt += n  # cumulative targets
            if torch.__version__[0] == "2":
                gi = gi.cpu().long()
                gj = gj.cpu().long()
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i].to(ps.device)
            # pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            # pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            giou = bbox_iou(
                pbox.T, tbox[i], x1y1x2y2=False, CIoU=True
            )  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(
                0
            ).type(
                tobj.dtype
            )  # giou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling
    lbox *= h["giou"] * s
    lobj *= h["obj"] * s * (1.4 if np == 4 else 1.0)
    lcls *= h["cls"] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    nt = targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)
    # normalized to gridspace gain
    off = torch.tensor(
        [[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device
    ).float()  # overlap offsets

    g = 0.5  # offset
    multi_gpu = is_parallel(model)
    for i, jj in enumerate(
        model.module.yolo_layers if multi_gpu else model.yolo_layers
    ):
        # get number of grid points and anchor vec for this yolo layer
        anchors = (
            model.module.module_list[jj].anchor_vec
            if multi_gpu
            else model.module_list[jj].anchor_vec
        )
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            na = anchors.shape[0]  # number of anchors
            if torch.__version__[0] == "2":
                at = (torch.arange(na).view(na, 1).repeat(1, nt)).to(
                    targets.device
                )  # anchor tensor, same as .repeat_interleave(nt)
            else:
                at = (
                    torch.arange(na).view(na, 1).repeat(1, nt).long()
                )  # anchor tensor, same as .repeat_interleave(nt)
            # print("predict")
            r = t[None, :, 4:6] / anchors[:, None].to(t.device)  # wh ratio
            j = torch.max(r, 1.0 / r).max(2)[0] < model.hyp["anchor_t"]  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
            l, m = ((gxy % 1.0 > (1 - g)) & (gxy < (gain[[2, 3]] - 1.0))).T
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat(
                (t, t[j], t[k], t[l], t[m]), 0
            )
            offsets = (
                torch.cat(
                    (z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0
                )
                * g
            )

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        # indices.append((b, a, gj, gi))  # image, anchor, grid indices
        if torch.__version__[0] == "2":
            indices.append(
                (
                    b,
                    a,
                    torch.max(
                        torch.zeros_like(gj.float()), torch.min(gj.float(), gain[3] - 1)
                    ),
                    torch.max(
                        torch.zeros_like(gi.float()), torch.min(gi.float(), gain[2] - 1)
                    ),
                )
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1).int())  # box
        else:
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box

        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    # w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    # w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    # union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-16  # add by me
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-16  # add by me
    union = (w1 * h1) + w2 * h2 - inter + 1e-16  # remove 1e-16

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_area = cw * ch + 1e-16  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
        # convex diagonal squared
        c2 = cw**2 + ch**2 + 1e-16
        # centerpoint distance squared
        rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + (
            (b2_y1 + b2_y2) - (b1_y1 + b1_y2)
        ) ** 2 / 4
        if DIoU:
            return iou - rho2 / c2  # DIoU
        v = (4 / math.pi**2) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
        )
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-16)
        return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou
