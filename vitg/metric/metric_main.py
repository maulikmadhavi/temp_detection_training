import torch
import numpy as np

from vitg.network.backbone.vitgyolov8.yolo.utils import ops as yolov8ops

# other functions for test
from vitg.utils.general import (
    ap_per_class,
    box_iou,
    clip_coords,
    non_max_suppression,
    xywh2xyxy,
)
from vitg.utils.torch_utils import time_synchronized


def calculate_train_map(
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
    save_dir="",
    class_num=1,
    merge=False,
    save_txt=False,
    config=None,
):
    # Initialize/load model and set device
    training = model is not None

    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:
        print(" not related ")  # called directly

    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    # with open(data) as f:
    #    data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    # nc = 1 if single_cls else int(data['nc'])  # number of classes
    nc = class_num
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    # try:
    #    names = model.names if hasattr(model, 'names') else model.module.names
    # except:
    #    names = load_classes(opt.names)
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
    stats, ap, ap_class = [], [], []

    # compute_loss_ota = ComputeLossAuxOTA(model)  # init loss class
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    for img, targets, paths, shapes in dataloader:
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
            inf_out, train_out = model(
                img, augment=augment
            )  # inference and training outputs
            t0 += time_synchronized() - t

            # Run NMS
            t = time_synchronized()
            # output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=False)
            output = (
                non_max_suppression(
                    inf_out,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    merge=False,
                )
                if config.arch != "yolov8"
                else yolov8ops.non_max_suppression(
                    inf_out,
                    conf_thres,
                    iou_thres,
                    labels=[],
                    multi_label=True,
                    agnostic=config.single_cls,
                    max_det=300,
                )
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

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
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

    # Print speeds
    t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (
        imgsz,
        imgsz,
        batch_size,
    )  # tuple
    if not training:
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
