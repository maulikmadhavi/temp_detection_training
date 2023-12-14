"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import statistics
import sys
import time
from enum import Enum

import numpy as np
import torch
import yaml
from torchvision.ops.boxes import box_convert, box_iou
from tqdm import tqdm


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """

    EveryPointInterpolation = 1
    ElevenPointInterpolation = 2


# In[26]:


class _BbFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH
    or (X1,Y1,X2,Y2) => XYX2Y2

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """

    XYWH = 1
    XYX2Y2 = 2


# In[34]:


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """

    Relative = 1
    Absolute = 2


# In[43]:


class _BbType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """

    GroundTruth = 1
    Detected = 2


class Evaluator:
    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1 + i] != mrec[i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0 : len(mpre) - 1], mrec[0 : len(mpre) - 1], ii]

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        inter_area = Evaluator._getIntersectionArea(boxA, boxB)
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=inter_area)
        # intersection over union
        iou = inter_area / union
        assert iou >= 0
        return iou

    @staticmethod
    def _boxes_intersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False
        if boxB[0] > boxA[2]:
            return False
        if boxA[3] < boxB[1]:
            return False
        return not boxA[1] > boxB[3]

    @staticmethod
    def _get_intersection_area(boxA, boxB):
        x_a = max(boxA[0], boxB[0])
        y_a = max(boxA[1], boxB[1])
        x_b = min(boxA[2], boxB[2])
        y_b = min(boxA[3], boxB[3])
        return (x_b - x_a + 1) * (y_b - y_a + 1)

    @staticmethod
    def _get_union_areas(boxA, boxB, interArea=None):
        area_a = Evaluator._getArea(boxA)
        area_b = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_a + area_b - interArea)

    @staticmethod
    def _get_area(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def encode_local(bboxes_in, labels_in, dboxes_default_encode, criteria=0.5):
    ious = box_iou(bboxes_in, dboxes_default_encode)
    best_dbox_ious, best_dbox_idx = ious.max(dim=0)
    _, best_bbox_idx = ious.max(dim=1)

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


def evaluate_mobilenet(
    opt,
    model,
    test_loader,
    epoch,
    writer,
    encoder,
    nms_threshold,
    criterion,
    dboxes_default,
):
    model.eval()
    detections = []
    allgroundt = []

    totalloss = 0
    for img, targets, paths, shapes in tqdm(test_loader, total=len(test_loader)):
        img = img.float() / 255.0
        time.time()
        # sample only for fast eval

        boxinallbatch = []
        labelinallbatch = []
        for batchidx in range(opt.batch_size):
            tempb = targets[targets[:, 0] == batchidx]
            temploc = tempb[:, 2:]

            templabel = tempb[:, 1] + 1
            temploc_updateform2 = [
                [
                    item[0][0] - item[0][2] / 2,
                    item[0][1] - item[0][3] / 2,
                    item[0][2],
                    item[0][3],
                    item[1],
                ]
                for item in zip(temploc, templabel)
            ]
            templocredotensor2 = torch.as_tensor(temploc_updateform2)

            temploc_updateform = [
                [
                    item[0] - item[2] / 2,
                    item[1] - item[3] / 2,
                    item[0] + item[2] / 2,
                    item[1] + item[3] / 2,
                ]
                for item in temploc
            ]
            templocredotensor = torch.as_tensor(temploc_updateform)

            bboxes, labels = encode_local(
                bboxes_in=templocredotensor,
                labels_in=templabel.to(torch.long),
                dboxes_default_encode=dboxes_default,
            )
            boxinallbatch.append(bboxes.numpy())
            labelinallbatch.append(labels.numpy())

            width = shapes[batchidx][0][1]
            height = shapes[batchidx][0][0]
            for value in templocredotensor2:
                allgroundt.append(
                    [
                        paths[batchidx],
                        value[0] * width,
                        value[1] * height,
                        value[2] * width,
                        value[3] * height,
                        1,
                        value[4],
                    ]
                )

        gloc = torch.tensor(boxinallbatch)
        glabel = torch.tensor(labelinallbatch)
        time.time()

        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()

        with torch.no_grad():
            # Get predictions
            plabel, ploc = model(img.to(torch.float))
            ploc, plabel = ploc.float(), plabel.float()
            gloc = gloc.transpose(1, 2).contiguous()

            ploc = ploc.view(opt.batch_size, -1, 3000)
            plabel = plabel.view(opt.batch_size, -1, 3000)

            loss = criterion(ploc.cpu(), plabel.cpu(), gloc.cpu(), glabel.cpu())

            detection_target_inbatch = []
            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[
                        0
                    ]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue
                width = shapes[idx][0][1]
                height = shapes[idx][0][0]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append(
                        [
                            paths[idx],
                            loc_[0] * width,
                            loc_[1] * height,
                            (loc_[2] - loc_[0]) * width,
                            (loc_[3] - loc_[1]) * height,
                            prob_,
                            label_,
                        ]
                    )
                    # category_ids[label_ - 1]])
                    if prob_ > 0.3:
                        detection_target_inbatch.append(
                            [
                                idx,
                                label_,
                                (loc_[0] + loc_[2]) / 2,
                                (loc_[3] + loc_[1]) / 2,
                                (loc_[2] - loc_[0]),
                                (loc_[3] - loc_[1]),
                            ]
                        )
                        # x1,y1,w,h

    totalloss += loss
    # hard code 0.05
    detections = [item for item in detections if item[-2] > 0.05]

    detections_aftercov = []
    for itemconv in detections:
        bboxconv = (
            float(itemconv[1]),
            float(itemconv[2]),
            float(itemconv[1]) + float(itemconv[3]),
            float(itemconv[2]) + float(itemconv[4]),
        )
        detections_aftercov.append(
            [
                str((itemconv[0])),
                str(float(itemconv[6])),
                float(itemconv[5]),
                bboxconv,
                "999",
            ]
        )

    allgroundt_aftercov = []
    for itemconv in allgroundt:
        bboxconv = (
            float(itemconv[1]),
            float(itemconv[2]),
            float(itemconv[1]) + float(itemconv[3]),
            float(itemconv[2]) + float(itemconv[4]),
        )
        allgroundt_aftercov.append(
            [
                str(itemconv[0]),
                str(float(itemconv[6])),
                float(itemconv[5]),
                bboxconv,
                "999",
            ]
        )

    detections_aftercov_class = [item[1] for item in detections_aftercov]
    allgroundt_aftercov_class = [item[1] for item in allgroundt_aftercov]
    detectionsgt_all_class = detections_aftercov_class + allgroundt_aftercov_class
    detectionsgt_all_class = list(set(detectionsgt_all_class))

    classesall = sorted(detectionsgt_all_class)

    iou_thresholdmin = 0.4
    iou_thresholdmax = 2
    method = MethodAveragePrecision.EveryPointInterpolation

    ret = (
        []
    )  # list containing metrics (precision, recall, average precision) of each class

    # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2),annotationid])
    ground_truths = allgroundt_aftercov
    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2),annotationid])
    detections = detections_aftercov
    # Get all classes
    classes = classesall
    # Loop through all bounding boxes and separate them into GTs and detections

    dects = []
    for c in classes:
        # Get only detection of class c

        # Get only ground truths of class c, use filename as key
        gts = {}
        npos = 0
        for g in ground_truths:
            if g[1] == c:
                npos += 1
                gts[g[0]] = gts.get(g[0], []) + [g]

        # sort detections by decreasing confidence
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
        TP = np.zeros(len(dects))
        TP_annoid = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with amount of gts for each image
        det = {key: np.zeros(len(gts[key])) for key in gts}
        # Loop through detections
        for d in range(len(dects)):
            # Find ground truth image
            gt = gts[dects[d][0]] if dects[d][0] in gts else []
            iouMax = sys.float_info.min
            for j in range(len(gt)):
                iou = Evaluator.iou(dects[d][3], gt[j][3])
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            # Assign detection as true positive/don't care/false positive
            if iouMax >= iou_thresholdmin and iouMax <= iou_thresholdmax:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive in this detection
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    TP_annoid[d] = gt[jmax][
                        -1
                    ]  # sotrt related annotation ID from annotation
                else:
                    FP[d] = 1  # count as false positive
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)

        # take all TP in the class if detection, with annoatation id
        dects_tp_fromidx = []
        for index in range(len(TP)):
            if TP[index] == 1:
                temp = dects[index]
                temp.append(
                    str(int(TP_annoid[index]))
                )  # add annotationID in this TP detection
                dects_tp_fromidx.append(temp)
        # take all FP in the class if detection, with annoatation id
        dects_FP_fromidx = [dects[index] for index in range(len(FP)) if FP[index] == 1]

        # select TP an FN from ground truth per class
        from_gt_TP = []
        from_gt_FN = []
        for gtkey in gts:
            gt_perkey = gts[gtkey]
            gt_result = det[gtkey]
            for gt_img_idx in range(len(gt_result)):
                if gt_result[gt_img_idx] == 1:
                    from_gt_TP.append(gt_perkey[gt_img_idx])
                elif gt_result[gt_img_idx] == 0:
                    from_gt_FN.append(gt_perkey[gt_img_idx])

        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        # Depending on the method, call the right implementation
        if method == MethodAveragePrecision.EveryPointInterpolation:
            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
        else:
            [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)
        # add class result in the dictionary to be returned
        r = {
            "class": c,
            "precision": prec,
            "recall": rec,
            "AP": ap,
            "interpolated precision": mpre,
            "interpolated recall": mrec,
            "total positives": npos,
            "total TP": np.sum(TP),
            "total FP": np.sum(FP),
            "total TP from gt": len(from_gt_TP),
            "total FN from gt": len(from_gt_FN),
            "all detail TP from detection": dects_tp_fromidx,
            "all detail FP from detection": dects_FP_fromidx,
            "all detail TP from ground truth": from_gt_TP,
            "all detail FN from ground truth": from_gt_FN,
        }
        ret.append(r)
    # as name all detail TP from detection, all detail FP from detection, all detail TP from ground truth, all detail FN from ground truth
    # are sore here per box per line in format [imageName,class,confidence, (bb coordinates),annotationid]

    all_pv = []
    all_rv = []
    all_a_pv = []

    for item in ret:
        try:
            precision = len(item["all detail TP from detection"]) / (
                len(item["all detail TP from detection"])
                + len(item["all detail FP from detection"])
            )
        except:
            precision = 0
        all_pv.append(precision)
        try:
            recall = len(item["all detail TP from ground truth"]) / (
                len(item["all detail TP from ground truth"])
                + len(item["all detail FN from ground truth"])
            )
        except:
            recall = 0
        all_rv.append(recall)
        all_a_pv.append(item["AP"])

    all_a_pv = [float(item) for item in all_a_pv]
    mean_ap = statistics.mean(all_a_pv)

    return totalloss, mean_ap


def evaluate_lmdb_outyaml_mobilenet(
    opt, model, test_loader, epoch, writer, encoder, nms_threshold, dboxes_default
):
    yaml_out_temp1 = {"info": {"dataset_id": "111ageval"}, "output_images": []}

    model.eval()
    detections = []

    dump_yaml_temp = []
    with open(opt.yaml_out_path, "w+") as f:
        f.write(yaml.dump(yaml_out_temp1))
        f.write("output:" + "\n")

        for img, targets, paths, shapes in test_loader:
            img = img.float() / 255.0
            if torch.cuda.is_available():
                img = img.cuda()
            with torch.no_grad():
                # Get predictions
                plabel, ploc = model(img.to(torch.float))
                ploc, plabel = ploc.float(), plabel.float()

                ploc = ploc.view(opt.batch_size, -1, 3000)
                plabel = plabel.view(opt.batch_size, -1, 3000)

                detection_target_inbatch = []
                for idx in range(ploc.shape[0]):
                    ploc_i = ploc[idx, :, :].unsqueeze(0)
                    plabel_i = plabel[idx, :, :].unsqueeze(0)
                    try:
                        result = encoder.decode_batch(
                            ploc_i, plabel_i, nms_threshold, 200
                        )[0]
                    except:
                        print("No object detected in idx: {}".format(idx))
                        continue
                    width = shapes[idx][0][1]
                    height = shapes[idx][0][0]
                    loc, label, prob = [r.cpu().numpy() for r in result]
                    annidx = 0
                    for loc_, label_, prob_ in zip(loc, label, prob):
                        annidx += 1
                        detections.append(
                            [
                                paths[idx],
                                loc_[0] * width,
                                loc_[1] * height,
                                (loc_[2] - loc_[0]) * width,
                                (loc_[3] - loc_[1]) * height,
                                prob_,
                                label_,
                            ]
                        )
                        # category_ids[label_ - 1]])

                        conf = str(prob_)
                        if float(conf) > 0.05:
                            gt = np.int(label_) - 1
                            temp_a = {
                                "input_id": str(paths[idx]),
                                "bbox": [
                                    float(loc_[0]),
                                    float(loc_[1]),
                                    float(loc_[2] - loc_[0]),
                                    float(loc_[3] - loc_[1]),
                                ],
                                "segmentation": [],
                                "points": [],
                                "category_id": [opt.category_dic[int(gt)]],
                                # "category_id": [str(gt)],
                                "category_confidence_value": [float(conf)],
                                # "category_confidence_value":[str(conf)],
                                "label": [],
                                "label_conf_value": [],
                                "output_image_id": "",
                                "output_id": "1" + str(annidx) + str(paths[idx]),
                            }
                            dump_yaml_temp.append(temp_a)

                        if prob_ > 0.3:
                            detection_target_inbatch.append(
                                [
                                    idx,
                                    label_,
                                    (loc_[0] + loc_[2]) / 2,
                                    (loc_[3] + loc_[1]) / 2,
                                    (loc_[2] - loc_[0]),
                                    (loc_[3] - loc_[1]),
                                ]
                            )

                    store_bef_write = yaml.dump(dump_yaml_temp)
                    if len(dump_yaml_temp) == 0:
                        pass

                    else:
                        f.write(store_bef_write)

                    dump_yaml_temp = []
