"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import json
import os
import shutil
import statistics

# import operator
import sys
import time
from enum import Enum
from glob import glob
from types import SimpleNamespace
import numpy as np
import pandas as pd

# from tqdm.autonotebook import tqdm
import torch
import yaml

# from pycocotools.cocoeval import COCOeval
# from apex import amp
from torchvision.ops.boxes import box_convert, box_iou

# sys.path.insert(1, '/home/alex/vade/VADE_detection_bitbuket/0.4.1_mobileSSD/src')
# from vitg.network.backbone.vitgyolor.utils.plots import plot_images

# import sys


# from vitg.stand_alone_metric import *


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


class BBFormat(Enum):
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


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """

    GroundTruth = 1
    Detected = 2


class Evaluator:
    # def GetPascalVOCMetrics(
    #     self,
    #     boundingboxes,
    #     IOUThreshold=0.5,
    #     method=MethodAveragePrecision.EveryPointInterpolation,
    # ):
    #     """Get the metrics used by the VOC Pascal 2012 challenge.
    #     Get
    #     Args:
    #         boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
    #         bounding boxes;
    #         IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
    #         (default value = 0.5);
    #         method (default = EveryPointInterpolation): It can be calculated as the implementation
    #         in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
    #         interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
    #         or EveryPointInterpolation"  (ElevenPointInterpolation);
    #     Returns:
    #         A list of dictionaries. Each dictionary contains information and metrics of each class.
    #         The keys of each dictionary are:
    #         dict['class']: class representing the current dictionary;
    #         dict['precision']: array with the precision values;
    #         dict['recall']: array with the recall values;
    #         dict['AP']: average precision;
    #         dict['interpolated precision']: interpolated precision values;
    #         dict['interpolated recall']: interpolated recall values;
    #         dict['total positives']: total number of ground truth positives;
    #         dict['total TP']: total number of True Positive detections;
    #         dict['total FP']: total number of False Positive detections;
    #     """
    #     ret = (
    #         []
    #     )  # list containing metrics (precision, recall, average precision) of each class
    #     # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
    #     groundTruths = []
    #     # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    #     detections = []
    #     # Get all classes
    #     classes = []
    #     # Loop through all bounding boxes and separate them into GTs and detections
    #     for bb in boundingboxes.getBoundingBoxes():
    #         # print(bb)
    #         # [imageName, class, confidence, (bb coordinates XYX2Y2)]
    #         if bb.getBBType() == BBType.GroundTruth:
    #             groundTruths.append(
    #                 [
    #                     bb.getImageName(),
    #                     bb.getClassId(),
    #                     1,
    #                     bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2),
    #                     bb.getannoName(),
    #                 ]
    #             )

    #         else:
    #             detections.append(
    #                 [
    #                     bb.getImageName(),
    #                     bb.getClassId(),
    #                     bb.getConfidence(),
    #                     bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2),
    #                     bb.getannoName(),
    #                 ]
    #             )
    #             print(
    #                 [
    #                     bb.getImageName(),
    #                     bb.getClassId(),
    #                     bb.getConfidence(),
    #                     bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2),
    #                     bb.getannoName(),
    #                 ]
    #             )
    #         # get class
    #         if bb.getClassId() not in classes:
    #             classes.append(bb.getClassId())
    #     classes = sorted(classes)
    #     # Precision x Recall is obtained individually by each class
    #     # Loop through by classes
    #     for c in classes:
    #         # Get only detection of class c
    #         dects = []
    #         [dects.append(d) for d in detections if d[1] == c]
    #         # Get only ground truths of class c, use filename as key
    #         gts = {}
    #         npos = 0
    #         for g in groundTruths:
    #             if g[1] == c:
    #                 npos += 1
    #                 gts[g[0]] = gts.get(g[0], []) + [g]

    #         # sort detections by decreasing confidence
    #         dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
    #         TP = np.zeros(len(dects))
    #         FP = np.zeros(len(dects))
    #         # create dictionary with amount of gts for each image
    #         det = {key: np.zeros(len(gts[key])) for key in gts}

    #         # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
    #         # Loop through detections
    #         for d in range(len(dects)):
    #             # print('dect %s => %s' % (dects[d][0], dects[d][3],))
    #             # Find ground truth image
    #             gt = gts[dects[d][0]] if dects[d][0] in gts else []
    #             iouMax = sys.float_info.min
    #             for j in range(len(gt)):
    #                 # print('Ground truth gt => %s' % (gt[j][3],))
    #                 iou = Evaluator.iou(dects[d][3], gt[j][3])
    #                 if iou > iouMax:
    #                     iouMax = iou
    #                     jmax = j
    #             # Assign detection as true positive/don't care/false positive
    #             if iouMax >= IOUThreshold:
    #                 if det[dects[d][0]][jmax] == 0:
    #                     TP[d] = 1  # count as true positive
    #                     det[dects[d][0]][jmax] = 1  # flag as already 'seen'
    #                     # print("TP")
    #                 else:
    #                     FP[d] = 1  # count as false positive
    #                     # print("FP")
    #             # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
    #             else:
    #                 FP[d] = 1  # count as false positive
    #                 # print("FP")
    #         # compute precision, recall and average precision
    #         acc_FP = np.cumsum(FP)
    #         acc_TP = np.cumsum(TP)
    #         rec = acc_TP / npos
    #         prec = np.divide(acc_TP, (acc_FP + acc_TP))
    #         # Depending on the method, call the right implementation
    #         if method == MethodAveragePrecision.EveryPointInterpolation:
    #             [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
    #         else:
    #             [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)
    #         # add class result in the dictionary to be returned
    #         r = {
    #             "class": c,
    #             "precision": prec,
    #             "recall": rec,
    #             "AP": ap,
    #             "interpolated precision": mpre,
    #             "interpolated recall": mrec,
    #             "total positives": npos,
    #             "total TP": np.sum(TP),
    #             "total FP": np.sum(FP),
    #         }
    #         ret.append(r)
    #     return ret

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

    # @staticmethod
    # # 11-point interpolated average precision
    # def ElevenPointInterpolatedAP(rec, prec):
    #     # def CalculateAveragePrecision2(rec, prec):
    #     mrec = []
    #     # mrec.append(0)
    #     [mrec.append(e) for e in rec]
    #     # mrec.append(1)
    #     mpre = []
    #     # mpre.append(0)
    #     [mpre.append(e) for e in prec]
    #     # mpre.append(0)
    #     recallValues = np.linspace(0, 1, 11)
    #     recallValues = list(recallValues[::-1])
    #     rhoInterp = []
    #     recallValid = []
    #     # For each recallValues (0, 0.1, 0.2, ... , 1)
    #     for r in recallValues:
    #         # Obtain all recall values higher or equal than r
    #         argGreaterRecalls = np.argwhere(mrec[:] >= r)
    #         pmax = 0
    #         # If there are recalls above r
    #         if argGreaterRecalls.size != 0:
    #             pmax = max(mpre[argGreaterRecalls.min() :])
    #         recallValid.append(r)
    #         rhoInterp.append(pmax)
    #     # By definition AP = sum(max(precision whose recall is above r))/11
    #     ap = sum(rhoInterp) / 11
    #     # Generating values for the plot
    #     rvals = []
    #     rvals.append(recallValid[0])
    #     [rvals.append(e) for e in recallValid]
    #     rvals.append(0)
    #     pvals = []
    #     pvals.append(0)
    #     [pvals.append(e) for e in rhoInterp]
    #     pvals.append(0)
    #     # rhoInterp = rhoInterp[::-1]
    #     cc = []
    #     for i in range(len(rvals)):
    #         p = (rvals[i], pvals[i - 1])
    #         if p not in cc:
    #             cc.append(p)
    #         p = (rvals[i], pvals[i])
    #         if p not in cc:
    #             cc.append(p)
    #     recallValues = [i[0] for i in cc]
    #     rhoInterp = [i[1] for i in cc]
    #     return [ap, rhoInterp, recallValues, None]

    # # For each detections, calculate IOU with reference
    # @staticmethod
    # def _getAllIOUs(reference, detections):
    #     ret = []
    #     bbReference = reference.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
    #     # img = np.zeros((200,200,3), np.uint8)
    #     for d in detections:
    #         bb = d.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
    #         iou = Evaluator.iou(bbReference, bb)
    #         # Show blank image with the bounding boxes
    #         # img = add_bb_into_image(img, d, color=(255,0,0), thickness=2, label=None)
    #         # img = add_bb_into_image(img, reference, color=(0,255,0), thickness=2, label=None)
    #         ret.append((iou, reference, d))  # iou, reference, detection
    #     # cv2.imshow("comparing",img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyWindow("comparing")
    #     return sorted(
    #         ret, key=lambda i: i[0], reverse=True
    #     )  # sort by iou (from highest to lowest)

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


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


# def train(
#     opt,
#     model,
#     train_loader,
#     epoch,
#     writer,
#     criterion,
#     optimizer,
#     scheduler,
#     is_amp,
#     encoder,
#     dboxes_default,
# ):
#     model.train()
#     num_iter_per_epoch = len(train_loader)
#     progress_bar = tqdm(train_loader)
#     # for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
#     for i, (img, targets, paths, shapes) in enumerate(progress_bar):
#         if opt.plot and i < 2:
#             f = f"train_batch{i}_labels.jpg"  # filename
#             targetsny = targets.numpy()
#             targetsny[:, 1] += 1
#             plot_images(
#                 img,
#                 targetsny,
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # labels
#         # print("shape")
#         # print(shapes)
#         # print("img size")
#         # print(img.shape)
#         # print("paths[batchidx]")
#         # print(paths[i])

#         boxinallbatch = []
#         labelinallbatch = []
#         for batchidx in range(opt.batch_size):
#             # print("image shape")
#             # print(shapes[batchidx])
#             # print(shapes[batchidx][0])
#             # print("new batch")
#             tempb = targets[targets[:, 0] == batchidx]
#             # print(tempb)
#             temploc = tempb[:, 2:]

#             templabel = tempb[:, 1] + 1
#             # print(temploc)
#             # print(templabel)

#             temploc_updateform = [
#                 [
#                     item[0] - item[2] / 2,
#                     item[1] - item[3] / 2,
#                     item[0] + item[2] / 2,
#                     item[1] + item[3] / 2,
#                 ]
#                 for item in temploc
#             ]
#             templocredotensor = torch.as_tensor(temploc_updateform)
#             # print(templocredotensor)
#             # print("default box shape")
#             # print(dboxes_default.shape)
#             bboxes, labels = encode_local(
#                 bboxes_in=templocredotensor,
#                 labels_in=templabel.to(torch.long),
#                 dboxes_default_encode=dboxes_default,
#             )
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             boxinallbatch.append(bboxes.numpy())
#             labelinallbatch.append(labels.numpy())

#         # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#         # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#         # result = torch.cat(outputs, dim=1)
#         gloc = torch.tensor(boxinallbatch)
#         glabel = torch.tensor(labelinallbatch)
#         # print("label not 0")
#         # print(glabel[glabel!=0])

#         # print("all tensor shape overall")
#         # print(boxinallbatchresult.shape)
#         # print(labelinallbatchresult.shape)

#         # print("img and label size")
#         # print(img.shape)
#         # print(gloc.shape)
#         # print(glabel.shape)

#         if torch.cuda.is_available():
#             img = img.cuda()
#             gloc = gloc.cuda()
#             glabel = glabel.cuda()

#         ploc, plabel = model(img.to(torch.float))
#         ploc, plabel = ploc.float(), plabel.float()
#         gloc = gloc.transpose(1, 2).contiguous()

#         loss = criterion(ploc, plabel, gloc, glabel)

#         progress_bar.set_description(
#             "Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item())
#         )

#         writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

#         if is_amp:
#             with amp.scale_loss(loss, optimizer) as scale_loss:
#                 scale_loss.backward()
#         else:
#             loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if opt.plot and (opt.epoch + 1) % 20 == 0 and i < 50:
#             nms_threshold = 0.5
#             # print("predict label")
#             # print(plabel)
#             detection_target_inbatch = []
#             for idx in range(ploc.shape[0]):
#                 ploc_i = ploc[idx, :, :].unsqueeze(0)
#                 plabel_i = plabel[idx, :, :].unsqueeze(0)
#                 try:
#                     # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                     result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 50)[
#                         0
#                     ]
#                 except:
#                     print("No object detected in idx: {}".format(idx))
#                     continue
#                 width = shapes[idx][0][1]
#                 height = shapes[idx][0][0]
#                 # height, width = img_size[idx]
#                 loc, label, prob = [r.cpu().detach().numpy() for r in result]
#                 for loc_, label_, prob_ in zip(loc, label, prob):
#                     # detections.append([paths[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
#                     #                    (loc_[3] - loc_[1]) * height, prob_,
#                     #                    label_])
#                     # category_ids[label_ - 1]])
#                     if prob_ > 0.3:
#                         detection_target_inbatch.append(
#                             [
#                                 idx,
#                                 label_,
#                                 (loc_[0] + loc_[2]) / 2,
#                                 (loc_[3] + loc_[1]) / 2,
#                                 (loc_[2] - loc_[0]),
#                                 (loc_[3] - loc_[1]),
#                             ]
#                         )

#             # print(paths)
#             f = f"train_batch{i}_labels.jpg"  # filename
#             targetsny = targets.numpy()
#             targetsny[:, 1] += 1
#             plot_images(
#                 img,
#                 targetsny,
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # labels
#             f = f"train_batch{i}_pred.jpg"
#             plot_images(
#                 img,
#                 torch.as_tensor(detection_target_inbatch),
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # predictions

#     scheduler.step()


# def evaluate_lmdb(
#     opt, model, test_loader, epoch, writer, encoder, nms_threshold, dboxes_default
# ):
#     if opt.dummy_val:
#         all_det_result = glob(
#             "/home/alex/vade/map_simple/mAP/input/detection-results/*.txt"
#         )
#         all_gt_result = glob("/home/alex/vade/map_simple/mAP/input/ground-truth/*.txt")

#         all_det_result_matrix = []
#         for lines in all_det_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_det_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[2]),
#                         float(indiline_split[3]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         float(indiline_split[5]) - float(indiline_split[3]),
#                         indiline_split[1],
#                         indiline_split[0],
#                     ]
#                 )
#         all_groundt_result_matrix = []
#         for lines in all_gt_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_groundt_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[1]),
#                         float(indiline_split[2]),
#                         float(indiline_split[3]) - float(indiline_split[1]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         1,
#                         indiline_split[0],
#                     ]
#                 )
#         caculate_mAP_local(all_det_result_matrix, all_groundt_result_matrix)

#     model.eval()
#     detections = []
#     allgroundt = []
#     # category_ids = test_loader.dataset.coco.getCatIds()
#     # for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

#     # print("hello")

#     # for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
#     for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
#         start = time.time()
#         # sample only for fast eval
#         # if nbatch>1:
#         #    break
#         # print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#         # print("batch")
#         # print
#         # print("shape")
#         # print(shapes)
#         # allgot_ccoform=[]

#         for batchidx in range(opt.batch_size):
#             # print("image shape")
#             # print(shapes[batchidx])
#             # print("new batch in xmid ymid w h")
#             tempb = targets[targets[:, 0] == batchidx]
#             # print(tempb)
#             temploc = tempb[:, 2:]

#             templabel = tempb[:, 1] + 1
#             # print(temploc)
#             # print(templabel)

#             # temploc_updateform2=[[item[2]-item[4]/2,item[3]-item[5]/2,item[4],item[5],item[1]] for item in zip(tempb,)]
#             temploc_updateform2 = [
#                 [
#                     item[0][0] - item[0][2] / 2,
#                     item[0][1] - item[0][3] / 2,
#                     item[0][2],
#                     item[0][3],
#                     item[1],
#                 ]
#                 for item in list(zip(temploc, templabel))
#             ]
#             templocredotensor2 = torch.as_tensor(temploc_updateform2)
#             # print("templocredotensor2 in x1 y1 w h")
#             # print(templocredotensor2)

#             # temploc_updateform=[[item[0]-item[2]/2,item[1]-item[3]/2,item[2],item[3]] for item in temploc]
#             ##templocredotensor = torch.as_tensor(temploc_updateform)
#             # print("templocredotensor")
#             # print(templocredotensor)
#             # bboxes, labels = encode_local(bboxes_in=templocredotensor, labels_in=templabel.to(torch.long))
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             # boxinallbatch.append(bboxes.numpy())
#             # labelinallbatch.append(labels.numpy())
#             width = shapes[batchidx][0][1]
#             height = shapes[batchidx][0][0]
#             for value in templocredotensor2:
#                 allgroundt.append(
#                     [
#                         paths[batchidx],
#                         value[0] * width,
#                         value[1] * height,
#                         value[2] * width,
#                         value[3] * height,
#                         1,
#                         value[4],
#                     ]
#                 )

#         # print("time in one gt loop")
#         # print(time.time()-start)
#         start = time.time()
#         # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#         # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#         # result = torch.cat(outputs, dim=1)
#         # boxinallbatchresult = torch.tensor(boxinallbatch)
#         # labelinallbatchresult = torch.tensor(labelinallbatch)

#         # print("all tensor shape overall")
#         # print(allgroundt)
#         # print(labelinallbatchresult.shape)

#         if torch.cuda.is_available():
#             img = img.cuda()
#         with torch.no_grad():
#             # Get predictions
#             ploc, plabel = model(img.to(torch.float))
#             ploc, plabel = ploc.float(), plabel.float()
#             # print("predict label")
#             # print(plabel)
#             detection_target_inbatch = []
#             for idx in range(ploc.shape[0]):
#                 ploc_i = ploc[idx, :, :].unsqueeze(0)
#                 plabel_i = plabel[idx, :, :].unsqueeze(0)
#                 try:
#                     # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                     result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[
#                         0
#                     ]
#                 except:
#                     print("No object detected in idx: {}".format(idx))
#                     continue
#                 width = shapes[idx][0][1]
#                 height = shapes[idx][0][0]
#                 # height, width = img_size[idx]
#                 loc, label, prob = [r.cpu().numpy() for r in result]
#                 for loc_, label_, prob_ in zip(loc, label, prob):
#                     detections.append(
#                         [
#                             paths[idx],
#                             loc_[0] * width,
#                             loc_[1] * height,
#                             (loc_[2] - loc_[0]) * width,
#                             (loc_[3] - loc_[1]) * height,
#                             prob_,
#                             label_,
#                         ]
#                     )
#                     # category_ids[label_ - 1]])
#                     if prob_ > 0.3:
#                         detection_target_inbatch.append(
#                             [
#                                 idx,
#                                 label_,
#                                 (loc_[0] + loc_[2]) / 2,
#                                 (loc_[3] + loc_[1]) / 2,
#                                 (loc_[2] - loc_[0]),
#                                 (loc_[3] - loc_[1]),
#                             ]
#                         )

#         # print("time in one nms loop")
#         # print(time.time()-start)

#         if nbatch < 50:
#             # print(paths)
#             f = f"test_batch{nbatch}_labels.jpg"  # filename
#             targetsny = targets.numpy()
#             targetsny[:, 1] += 1
#             plot_images(
#                 img,
#                 targetsny,
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # labels
#             f = f"test_batch{nbatch}_pred.jpg"
#             plot_images(
#                 img,
#                 torch.as_tensor(detection_target_inbatch),
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # predictions

#     detections = np.array(detections, dtype=np.float32)
#     allgroundt = np.array(allgroundt, dtype=np.float32)

#     # print("detect")
#     detections_class = [item[-1] for item in detections]
#     detections_class_u = list(set(detections_class))
#     print("all detect class")
#     print(detections_class_u)
#     allgroundt_class = [item[-1] for item in allgroundt]
#     allgroundt_class_u = list(set(allgroundt_class))
#     print("all gt class")
#     print(allgroundt_class_u)

#     # print(detections[0])
#     # print("allgroundt")
#     # print(allgroundt[0])

#     caculate_mAP_local(detections, allgroundt)

#     # coco_eval = COCOeval(allgroundt, detections, iouType="bbox")
#     # coco_eval = COCOeval(test_loader.dataset.coco.loadRes(allgroundt), test_loader.dataset.coco.loadRes(detections), iouType="bbox")
#     # coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
#     # coco_eval.evaluate()
#     # coco_eval.accumulate()
#     # coco_eval.summarize()

#     # writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)


# def evaluate_lmdb_inter(
#     opt, model, test_loader, epoch, writer, encoder, nms_threshold, dboxes_default
# ):
#     if opt.dummy_val:
#         all_det_result = glob(
#             "/home/alex/vade/map_simple/mAP/input/detection-results/*.txt"
#         )
#         all_gt_result = glob("/home/alex/vade/map_simple/mAP/input/ground-truth/*.txt")

#         all_det_result_matrix = []
#         for lines in all_det_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_det_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[2]),
#                         float(indiline_split[3]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         float(indiline_split[5]) - float(indiline_split[3]),
#                         indiline_split[1],
#                         indiline_split[0],
#                     ]
#                 )
#         all_groundt_result_matrix = []
#         for lines in all_gt_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_groundt_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[1]),
#                         float(indiline_split[2]),
#                         float(indiline_split[3]) - float(indiline_split[1]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         1,
#                         indiline_split[0],
#                     ]
#                 )
#         caculate_mAP_local(all_det_result_matrix, all_groundt_result_matrix)

#     model.eval()
#     detections = []
#     allgroundt = []
#     # category_ids = test_loader.dataset.coco.getCatIds()
#     # for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

#     # print("hello")

#     # for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
#     for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
#         img = img.float() / 255.0
#         start = time.time()
#         # sample only for fast eval
#         # if nbatch>1:
#         #    break
#         # print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#         # print("batch")
#         # print
#         # print("shape")
#         # print(shapes)
#         # allgot_ccoform=[]

#         for batchidx in range(opt.batch_size):
#             # print("image shape")
#             # print(shapes[batchidx])
#             # print("new batch in xmid ymid w h")
#             tempb = targets[targets[:, 0] == batchidx]
#             # print(tempb)
#             temploc = tempb[:, 2:]

#             templabel = tempb[:, 1] + 1
#             # print(temploc)
#             # print(templabel)

#             # temploc_updateform2=[[item[2]-item[4]/2,item[3]-item[5]/2,item[4],item[5],item[1]] for item in zip(tempb,)]
#             temploc_updateform2 = [
#                 [
#                     item[0][0] - item[0][2] / 2,
#                     item[0][1] - item[0][3] / 2,
#                     item[0][2],
#                     item[0][3],
#                     item[1],
#                 ]
#                 for item in list(zip(temploc, templabel))
#             ]
#             templocredotensor2 = torch.as_tensor(temploc_updateform2)
#             # print("templocredotensor2 in x1 y1 w h")
#             # print(templocredotensor2)

#             # temploc_updateform=[[item[0]-item[2]/2,item[1]-item[3]/2,item[2],item[3]] for item in temploc]
#             ##templocredotensor = torch.as_tensor(temploc_updateform)
#             # print("templocredotensor")
#             # print(templocredotensor)
#             # bboxes, labels = encode_local(bboxes_in=templocredotensor, labels_in=templabel.to(torch.long))
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             # boxinallbatch.append(bboxes.numpy())
#             # labelinallbatch.append(labels.numpy())
#             width = shapes[batchidx][0][1]
#             height = shapes[batchidx][0][0]
#             for value in templocredotensor2:
#                 allgroundt.append(
#                     [
#                         paths[batchidx],
#                         value[0] * width,
#                         value[1] * height,
#                         value[2] * width,
#                         value[3] * height,
#                         1,
#                         value[4],
#                     ]
#                 )

#         # print("time in one gt loop")
#         # print(time.time()-start)
#         start = time.time()
#         # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#         # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#         # result = torch.cat(outputs, dim=1)
#         # boxinallbatchresult = torch.tensor(boxinallbatch)
#         # labelinallbatchresult = torch.tensor(labelinallbatch)

#         # print("all tensor shape overall")
#         # print(allgroundt)
#         # print(labelinallbatchresult.shape)

#         if torch.cuda.is_available():
#             img = img.cuda()
#         with torch.no_grad():
#             # Get predictions
#             ploc, plabel = model(img.to(torch.float))
#             ploc, plabel = ploc.float(), plabel.float()
#             # print("predict label")
#             # print(plabel)
#             detection_target_inbatch = []
#             for idx in range(ploc.shape[0]):
#                 ploc_i = ploc[idx, :, :].unsqueeze(0)
#                 plabel_i = plabel[idx, :, :].unsqueeze(0)
#                 try:
#                     # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                     result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[
#                         0
#                     ]
#                 except:
#                     print("No object detected in idx: {}".format(idx))
#                     continue
#                 width = shapes[idx][0][1]
#                 height = shapes[idx][0][0]
#                 # height, width = img_size[idx]
#                 loc, label, prob = [r.cpu().numpy() for r in result]
#                 for loc_, label_, prob_ in zip(loc, label, prob):
#                     detections.append(
#                         [
#                             paths[idx],
#                             loc_[0] * width,
#                             loc_[1] * height,
#                             (loc_[2] - loc_[0]) * width,
#                             (loc_[3] - loc_[1]) * height,
#                             prob_,
#                             label_,
#                         ]
#                     )
#                     # category_ids[label_ - 1]])
#                     if prob_ > 0.3:
#                         detection_target_inbatch.append(
#                             [
#                                 idx,
#                                 label_,
#                                 (loc_[0] + loc_[2]) / 2,
#                                 (loc_[3] + loc_[1]) / 2,
#                                 (loc_[2] - loc_[0]),
#                                 (loc_[3] - loc_[1]),
#                             ]
#                         )

#         # print("time in one nms loop")
#         # print(time.time()-start)

#         if nbatch < 50:
#             # print(paths)
#             f = f"test_batch{nbatch}_labels.jpg"  # filename
#             targetsny = targets.numpy()
#             targetsny[:, 1] += 1
#             plot_images(
#                 img,
#                 targetsny,
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # labels
#             f = f"test_batch{nbatch}_pred.jpg"
#             plot_images(
#                 img,
#                 torch.as_tensor(detection_target_inbatch),
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # predictions

#     detections = np.array(detections, dtype=np.float32)
#     allgroundt = np.array(allgroundt, dtype=np.float32)

#     # print("detect")
#     detections_class = [item[-1] for item in detections]
#     detections_class_u = list(set(detections_class))
#     print("all detect class")
#     print(detections_class_u)
#     allgroundt_class = [item[-1] for item in allgroundt]
#     allgroundt_class_u = list(set(allgroundt_class))
#     print("all gt class")
#     print(allgroundt_class_u)

#     # print(detections[0])
#     # print("allgroundt")
#     # print(allgroundt[0])

#     caculate_mAP_local(detections, allgroundt)

#     # coco_eval = COCOeval(allgroundt, detections, iouType="bbox")
#     # coco_eval = COCOeval(test_loader.dataset.coco.loadRes(allgroundt), test_loader.dataset.coco.loadRes(detections), iouType="bbox")
#     # coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
#     # coco_eval.evaluate()
#     # coco_eval.accumulate()
#     # coco_eval.summarize()

#     # writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)


# def evaluate_lmdb_returnloss_v3(
#     opt,
#     model,
#     test_loader,
#     epoch,
#     writer,
#     encoder,
#     nms_threshold,
#     criterion,
#     dboxes_default,
# ):
#     model.eval()
#     detections = []
#     allgroundt = []
#     # category_ids = test_loader.dataset.coco.getCatIds()
#     # for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

#     # print("hello")
#     totalloss = 0
#     # for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
#     for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
#         img = img.float() / 255.0
#         # if nbatch>2:
#         #    break
#         start = time.time()
#         # sample only for fast eval
#         # if nbatch>1:
#         #    break
#         # print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#         # print("batch")
#         # print
#         # print("shape")
#         # print(shapes)
#         # allgot_ccoform=[]
#         # print("all tensor shape overall")
#         # print(boxinallbatchresult.shape)
#         # print(labelinallbatchresult.shape)

#         # print("img and label size")
#         # print(img.shape)
#         # print(gloc.shape)
#         # print(glabel.shape)

#         boxinallbatch = []
#         labelinallbatch = []
#         for batchidx in range(opt.batch_size):
#             # print("image shape")
#             # print(shapes[batchidx])
#             # print("new batch in xmid ymid w h")
#             tempb = targets[targets[:, 0] == batchidx]
#             # print(tempb)
#             temploc = tempb[:, 2:]

#             templabel = tempb[:, 1] + 1
#             # print(temploc)
#             # print(templabel)

#             # temploc_updateform2=[[item[2]-item[4]/2,item[3]-item[5]/2,item[4],item[5],item[1]] for item in zip(tempb,)]
#             temploc_updateform2 = [
#                 [
#                     item[0][0] - item[0][2] / 2,
#                     item[0][1] - item[0][3] / 2,
#                     item[0][2],
#                     item[0][3],
#                     item[1],
#                 ]
#                 for item in list(zip(temploc, templabel))
#             ]
#             templocredotensor2 = torch.as_tensor(temploc_updateform2)
#             # print("templocredotensor2 in x1 y1 w h")
#             # print(templocredotensor2)

#             temploc_updateform = [
#                 [
#                     item[0] - item[2] / 2,
#                     item[1] - item[3] / 2,
#                     item[0] + item[2] / 2,
#                     item[1] + item[3] / 2,
#                 ]
#                 for item in temploc
#             ]
#             templocredotensor = torch.as_tensor(temploc_updateform)
#             # print(templocredotensor)
#             # print("default box shape")
#             # print(dboxes_default.shape)
#             bboxes, labels = encode_local(
#                 bboxes_in=templocredotensor,
#                 labels_in=templabel.to(torch.long),
#                 dboxes_default_encode=dboxes_default,
#             )
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             boxinallbatch.append(bboxes.numpy())
#             labelinallbatch.append(labels.numpy())

#             # temploc_updateform=[[item[0]-item[2]/2,item[1]-item[3]/2,item[2],item[3]] for item in temploc]
#             ##templocredotensor = torch.as_tensor(temploc_updateform)
#             # print("templocredotensor")
#             # print(templocredotensor)
#             # bboxes, labels = encode_local(bboxes_in=templocredotensor, labels_in=templabel.to(torch.long))
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             # boxinallbatch.append(bboxes.numpy())
#             # labelinallbatch.append(labels.numpy())
#             width = shapes[batchidx][0][1]
#             height = shapes[batchidx][0][0]
#             for value in templocredotensor2:
#                 allgroundt.append(
#                     [
#                         paths[batchidx],
#                         value[0] * width,
#                         value[1] * height,
#                         value[2] * width,
#                         value[3] * height,
#                         1,
#                         value[4],
#                     ]
#                 )

#         gloc = torch.tensor(boxinallbatch)
#         glabel = torch.tensor(labelinallbatch)
#         # print("time in one gt loop")
#         # print(time.time()-start)
#         start = time.time()
#         # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#         # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#         # result = torch.cat(outputs, dim=1)
#         # boxinallbatchresult = torch.tensor(boxinallbatch)
#         # labelinallbatchresult = torch.tensor(labelinallbatch)

#         # print("all tensor shape overall")
#         # print(allgroundt)
#         # print(labelinallbatchresult.shape)

#         if torch.cuda.is_available():
#             img = img.cuda()
#             gloc = gloc.cuda()
#             glabel = glabel.cuda()

#         # ploc, plabel = model(img.to(torch.float))
#         # ploc, plabel = ploc.float(), plabel.float()

#         with torch.no_grad():
#             # Get predictions
#             # ploc, plabel = model(img.to(torch.float))
#             # ploc, plabel = ploc.float(), plabel.float()
#             # gloc = gloc.transpose(1, 2).contiguous()

#             plabel, ploc = model(img.to(torch.float))
#             ploc, plabel = ploc.float(), plabel.float()
#             gloc = gloc.transpose(1, 2).contiguous()

#             ploc = ploc.view(opt.batch_size, -1, 3000)
#             plabel = plabel.view(opt.batch_size, -1, 3000)

#             loss = criterion(ploc.cpu(), plabel.cpu(), gloc.cpu(), glabel.cpu())

#             # print("predict label")
#             # print(plabel)
#             detection_target_inbatch = []
#             for idx in range(ploc.shape[0]):
#                 ploc_i = ploc[idx, :, :].unsqueeze(0)
#                 plabel_i = plabel[idx, :, :].unsqueeze(0)
#                 try:
#                     # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                     result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[
#                         0
#                     ]
#                 except:
#                     print("No object detected in idx: {}".format(idx))
#                     continue
#                 width = shapes[idx][0][1]
#                 height = shapes[idx][0][0]
#                 # height, width = img_size[idx]
#                 loc, label, prob = [r.cpu().numpy() for r in result]
#                 for loc_, label_, prob_ in zip(loc, label, prob):
#                     detections.append(
#                         [
#                             paths[idx],
#                             loc_[0] * width,
#                             loc_[1] * height,
#                             (loc_[2] - loc_[0]) * width,
#                             (loc_[3] - loc_[1]) * height,
#                             prob_,
#                             label_,
#                         ]
#                     )
#                     # category_ids[label_ - 1]])
#                     if prob_ > 0.3:
#                         detection_target_inbatch.append(
#                             [
#                                 idx,
#                                 label_,
#                                 (loc_[0] + loc_[2]) / 2,
#                                 (loc_[3] + loc_[1]) / 2,
#                                 (loc_[2] - loc_[0]),
#                                 (loc_[3] - loc_[1]),
#                             ]
#                         )

#         # print("time in one nms loop")
#         # print(time.time()-start)

#         # if nbatch < 50:
#         if opt.dummy_plot and nbatch < 20:
#             # if opt.dummy_plot and (epoch+1)%10==0 and nbatch < 20:
#             # print(paths)
#             f = f"test_batch{nbatch}_labels.jpg"  # filename
#             targetsny = targets.numpy()
#             targetsny[:, 1] += 1
#             try:
#                 os.remove(f)
#                 plot_images(
#                     img,
#                     targetsny,
#                     [str(item) for item in paths],
#                     f,
#                     [str(item) for item in range(100)],
#                 )  # labels
#             except:
#                 plot_images(
#                     img,
#                     targetsny,
#                     [str(item) for item in paths],
#                     f,
#                     [str(item) for item in range(100)],
#                 )  # labels
#             f = f"test_batch{nbatch}_pred.jpg"
#             try:
#                 os.remove(f)
#                 plot_images(
#                     img,
#                     torch.as_tensor(detection_target_inbatch),
#                     [str(item) for item in paths],
#                     f,
#                     [str(item) for item in range(100)],
#                 )  # predictions
#             except:
#                 plot_images(
#                     img,
#                     torch.as_tensor(detection_target_inbatch),
#                     [str(item) for item in paths],
#                     f,
#                     [str(item) for item in range(100)],
#                 )  # predictions

#     totalloss += loss
#     # hard code 0.05
#     print("before length")
#     print(len(detections))
#     # print(detections[:3])
#     detections_score = [item[-2] for item in detections]
#     print("min score")
#     print(min(detections_score))
#     print("max score")
#     print(max(detections_score))
#     detections = [item for item in detections if item[-2] > 0.1]

#     print("after length")
#     print(len(detections))

#     detections = np.array(detections, dtype=np.float32)
#     allgroundt = np.array(allgroundt, dtype=np.float32)

#     print("detect")
#     detections_class = [item[-1] for item in detections]
#     detections_class_u = list(set(detections_class))
#     print("all detect class")
#     print(detections_class_u)
#     allgroundt_class = [item[-1] for item in allgroundt]
#     allgroundt_class_u = list(set(allgroundt_class))
#     print("all gt class")
#     print(allgroundt_class_u)

#     # print(detections[0])
#     # print("allgroundt")
#     # print(allgroundt[0])

#     allclassmap = caculate_mAP_local(detections, allgroundt)

#     return totalloss, allclassmap


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

    # print("hello")
    totalloss = 0
    for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
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
                for item in list(zip(temploc, templabel))
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

            # print("predict label")
            # print(plabel)
            detection_target_inbatch = []
            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[
                        0
                    ]
                except:
                    print("No object detected in idx: {}".format(idx))
                    continue
                width = shapes[idx][0][1]
                height = shapes[idx][0][0]
                # height, width = img_size[idx]
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

    Evaluator()

    # print("start calculate ap with boxes")
    # detections = evaluator.PlotPrecisionRecallCurve(
    # allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
    # IOUThreshold=iouThreshold,  # IOU threshold
    # method=MethodAveragePrecision.EveryPointInterpolation,
    # showAP=True,  # Show Average Precision in the title of the plot
    # showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
    # savePath="/home/alex/VADE/detection_metric/Object-Detection-Metrics/results",
    # showGraphic=False)

    # boundingboxes=allBoundingBoxes

    IOUThresholdmin = 0.4
    IOUThresholdmax = 2
    method = MethodAveragePrecision.EveryPointInterpolation

    ret = (
        []
    )  # list containing metrics (precision, recall, average precision) of each class

    # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2),annotationid])
    groundTruths = allgroundt_aftercov
    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2),annotationid])
    detections = detections_aftercov
    # Get all classes
    classes = classesall
    # Loop through all bounding boxes and separate them into GTs and detections

    # print("sample groundTruths")
    # print(len(groundTruths))
    # print(groundTruths[0])

    # print("sample detections")
    # print(len(detections))
    # print(detections[0])

    # print("all class")
    # print(classes)

    # print("start to calculate all metic for all classes")
    for c in classes:
        # print("class")
        # print(c)
        # Get only detection of class c
        dects = []
        [dects.append(d) for d in detections if d[1] == c]
        # Get only ground truths of class c, use filename as key
        gts = {}
        npos = 0
        for g in groundTruths:
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

        # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # print(dects)
        # Loop through detections
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))
            # Find ground truth image
            gt = gts[dects[d][0]] if dects[d][0] in gts else []
            iouMax = sys.float_info.min
            for j in range(len(gt)):
                # print('Ground truth gt => %s' % (gt[j][3],))
                # print(gt[j])
                iou = Evaluator.iou(dects[d][3], gt[j][3])
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            # Assign detection as true positive/don't care/false positive
            if iouMax >= IOUThresholdmin and iouMax <= IOUThresholdmax:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive in this detection
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    TP_annoid[d] = gt[jmax][
                        -1
                    ]  # sotrt related annotation ID from annotation
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
                # print("FP")
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
                # dects_tp_fromidx.append(dects[index])

        # take all FP in the class if detection, with annoatation id
        dects_FP_fromidx = []
        for index in range(len(FP)):
            if FP[index] == 1:
                dects_FP_fromidx.append(dects[index])

        # select TP an FN from ground truth per class
        from_gt_TP = []
        from_gt_FN = []
        for gtkey in gts:
            # print(gtkey)
            gt_perkey = gts[gtkey]
            gt_result = det[gtkey]
            # print(gt_perkey)
            # print(gt_result)
            for gt_img_idx in range(len(gt_result)):
                if gt_result[gt_img_idx] == 1:
                    from_gt_TP.append(gt_perkey[gt_img_idx])
                elif gt_result[gt_img_idx] == 0:
                    from_gt_FN.append(gt_perkey[gt_img_idx])

                    # print("detection TP, FP")
        # print(len(dects_tp_fromidx))
        # print(len(dects_FP_fromidx))

        # print("gt TP, FP")
        # print(len(from_gt_TP))
        # print(len(from_gt_FN))

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

    allPv = []
    allRv = []
    allAPv = []

    for item in ret:
        try:
            precision = len(item["all detail TP from detection"]) / (
                len(item["all detail TP from detection"])
                + len(item["all detail FP from detection"])
            )
        except:
            precision = 0
        allPv.append(precision)
        # recall=item["total TP"]/item["total positives"]
        try:
            recall = len(item["all detail TP from ground truth"]) / (
                len(item["all detail TP from ground truth"])
                + len(item["all detail FN from ground truth"])
            )
        except:
            recall = 0
        allRv.append(recall)
        allAPv.append(item["AP"])

    allAPv = [float(item) for item in allAPv]
    mean_ap = statistics.mean(allAPv)

    return totalloss, mean_ap


# def evaluate_lmdb_returnloss_v4_updatemetirc(
#     opt,
#     model,
#     test_loader,
#     epoch,
#     writer,
#     encoder,
#     nms_threshold,
#     criterion,
#     dboxes_default,
# ):
#     model.eval()
#     detections = []
#     allgroundt = []
#     # category_ids = test_loader.dataset.coco.getCatIds()
#     # for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

#     # print("hello")
#     totalloss = 0
#     # for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
#     for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
#         img = img.float() / 255.0
#         # if nbatch>2:
#         #    break
#         start = time.time()
#         # sample only for fast eval
#         # if nbatch>1:
#         #    break
#         # print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#         # print("batch")
#         # print
#         # print("shape")
#         # print(shapes)
#         # allgot_ccoform=[]
#         # print("all tensor shape overall")
#         # print(boxinallbatchresult.shape)
#         # print(labelinallbatchresult.shape)

#         # print("img and label size")
#         # print(img.shape)
#         # print(gloc.shape)
#         # print(glabel.shape)

#         boxinallbatch = []
#         labelinallbatch = []
#         for batchidx in range(opt.batch_size):
#             # print("image shape")
#             # print(shapes[batchidx])
#             # print("new batch in xmid ymid w h")
#             tempb = targets[targets[:, 0] == batchidx]
#             # print(tempb)
#             temploc = tempb[:, 2:]

#             templabel = tempb[:, 1] + 1
#             # print(temploc)
#             # print(templabel)

#             # temploc_updateform2=[[item[2]-item[4]/2,item[3]-item[5]/2,item[4],item[5],item[1]] for item in zip(tempb,)]
#             temploc_updateform2 = [
#                 [
#                     item[0][0] - item[0][2] / 2,
#                     item[0][1] - item[0][3] / 2,
#                     item[0][2],
#                     item[0][3],
#                     item[1],
#                 ]
#                 for item in list(zip(temploc, templabel))
#             ]
#             templocredotensor2 = torch.as_tensor(temploc_updateform2)
#             # print("templocredotensor2 in x1 y1 w h")
#             # print(templocredotensor2)

#             temploc_updateform = [
#                 [
#                     item[0] - item[2] / 2,
#                     item[1] - item[3] / 2,
#                     item[0] + item[2] / 2,
#                     item[1] + item[3] / 2,
#                 ]
#                 for item in temploc
#             ]
#             templocredotensor = torch.as_tensor(temploc_updateform)
#             # print(templocredotensor)
#             # print("default box shape")
#             # print(dboxes_default.shape)
#             bboxes, labels = encode_local(
#                 bboxes_in=templocredotensor,
#                 labels_in=templabel.to(torch.long),
#                 dboxes_default_encode=dboxes_default,
#             )
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             boxinallbatch.append(bboxes.numpy())
#             labelinallbatch.append(labels.numpy())

#             # temploc_updateform=[[item[0]-item[2]/2,item[1]-item[3]/2,item[2],item[3]] for item in temploc]
#             ##templocredotensor = torch.as_tensor(temploc_updateform)
#             # print("templocredotensor")
#             # print(templocredotensor)
#             # bboxes, labels = encode_local(bboxes_in=templocredotensor, labels_in=templabel.to(torch.long))
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             # boxinallbatch.append(bboxes.numpy())
#             # labelinallbatch.append(labels.numpy())
#             width = shapes[batchidx][0][1]
#             height = shapes[batchidx][0][0]
#             for value in templocredotensor2:
#                 allgroundt.append(
#                     [
#                         paths[batchidx],
#                         value[0] * width,
#                         value[1] * height,
#                         value[2] * width,
#                         value[3] * height,
#                         1,
#                         value[4],
#                     ]
#                 )

#         gloc = torch.tensor(boxinallbatch)
#         glabel = torch.tensor(labelinallbatch)
#         # print("time in one gt loop")
#         # print(time.time()-start)
#         start = time.time()
#         # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#         # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#         # result = torch.cat(outputs, dim=1)
#         # boxinallbatchresult = torch.tensor(boxinallbatch)
#         # labelinallbatchresult = torch.tensor(labelinallbatch)

#         # print("all tensor shape overall")
#         # print(allgroundt)
#         # print(labelinallbatchresult.shape)

#         if torch.cuda.is_available():
#             img = img.cuda()
#             gloc = gloc.cuda()
#             glabel = glabel.cuda()

#         # ploc, plabel = model(img.to(torch.float))
#         # ploc, plabel = ploc.float(), plabel.float()

#         with torch.no_grad():
#             # Get predictions
#             # ploc, plabel = model(img.to(torch.float))
#             # ploc, plabel = ploc.float(), plabel.float()
#             # gloc = gloc.transpose(1, 2).contiguous()

#             plabel, ploc = model(img.to(torch.float))
#             ploc, plabel = ploc.float(), plabel.float()
#             gloc = gloc.transpose(1, 2).contiguous()

#             ploc = ploc.view(opt.batch_size, -1, 3000)
#             plabel = plabel.view(opt.batch_size, -1, 3000)

#             loss = criterion(ploc.cpu(), plabel.cpu(), gloc.cpu(), glabel.cpu())

#             # print("predict label")
#             # print(plabel)
#             detection_target_inbatch = []
#             for idx in range(ploc.shape[0]):
#                 ploc_i = ploc[idx, :, :].unsqueeze(0)
#                 plabel_i = plabel[idx, :, :].unsqueeze(0)
#                 try:
#                     # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                     result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[
#                         0
#                     ]
#                 except:
#                     print("No object detected in idx: {}".format(idx))
#                     continue
#                 width = shapes[idx][0][1]
#                 height = shapes[idx][0][0]
#                 # height, width = img_size[idx]
#                 loc, label, prob = [r.cpu().numpy() for r in result]
#                 for loc_, label_, prob_ in zip(loc, label, prob):
#                     detections.append(
#                         [
#                             paths[idx],
#                             loc_[0] * width,
#                             loc_[1] * height,
#                             (loc_[2] - loc_[0]) * width,
#                             (loc_[3] - loc_[1]) * height,
#                             prob_,
#                             label_,
#                         ]
#                     )
#                     # category_ids[label_ - 1]])
#                     if prob_ > 0.3:
#                         detection_target_inbatch.append(
#                             [
#                                 idx,
#                                 label_,
#                                 (loc_[0] + loc_[2]) / 2,
#                                 (loc_[3] + loc_[1]) / 2,
#                                 (loc_[2] - loc_[0]),
#                                 (loc_[3] - loc_[1]),
#                             ]
#                         )
#                         # x1,y1,w,h
#         # print("time in one nms loop")
#         # print(time.time()-start)

#         # if nbatch < 50:
#         if opt.dummy_plot and nbatch < 20:
#             # if opt.dummy_plot and (epoch+1)%10==0 and nbatch < 20:
#             # print(paths)
#             f = f"test_batch{nbatch}_labels.jpg"  # filename
#             targetsny = targets.numpy()
#             targetsny[:, 1] += 1
#             try:
#                 os.remove(f)
#                 plot_images(
#                     img,
#                     targetsny,
#                     [str(item) for item in paths],
#                     f,
#                     [str(item) for item in range(100)],
#                 )  # labels
#             except:
#                 plot_images(
#                     img,
#                     targetsny,
#                     [str(item) for item in paths],
#                     f,
#                     [str(item) for item in range(100)],
#                 )  # labels
#             f = f"test_batch{nbatch}_pred.jpg"
#             try:
#                 os.remove(f)
#                 plot_images(
#                     img,
#                     torch.as_tensor(detection_target_inbatch),
#                     [str(item) for item in paths],
#                     f,
#                     [str(item) for item in range(100)],
#                 )  # predictions
#             except:
#                 plot_images(
#                     img,
#                     torch.as_tensor(detection_target_inbatch),
#                     [str(item) for item in paths],
#                     f,
#                     [str(item) for item in range(100)],
#                 )  # predictions

#     totalloss += loss
#     # hard code 0.05
#     # print("before length")
#     # print(len(detections))
#     # print(detections[:3])
#     detections_score = [item[-2] for item in detections]
#     # print("min score")
#     # print(min(detections_score))
#     # print("max score")
#     # print(max(detections_score))
#     # if opt.arch == "mobilenetssd":
#     detections = [item for item in detections if item[-2] > 0.05]

#     # print("after length")
#     # print(len(detections))
#     # print("all GTS")
#     # print(len(allgroundt))

#     # print("detection samples")
#     # print(detections[0])
#     # print("gt samples")
#     # print(allgroundt[0])

#     detections_aftercov = []
#     for itemconv in detections:
#         bboxconv = (
#             float(itemconv[1]),
#             float(itemconv[2]),
#             float(itemconv[1]) + float(itemconv[3]),
#             float(itemconv[2]) + float(itemconv[4]),
#         )
#         detections_aftercov.append(
#             [
#                 str((itemconv[0])),
#                 str(float(itemconv[6])),
#                 float(itemconv[5]),
#                 bboxconv,
#                 "999",
#             ]
#         )

#     allgroundt_aftercov = []
#     for itemconv in allgroundt:
#         bboxconv = (
#             float(itemconv[1]),
#             float(itemconv[2]),
#             float(itemconv[1]) + float(itemconv[3]),
#             float(itemconv[2]) + float(itemconv[4]),
#         )
#         allgroundt_aftercov.append(
#             [
#                 str(itemconv[0]),
#                 str(float(itemconv[6])),
#                 float(itemconv[5]),
#                 bboxconv,
#                 "999",
#             ]
#         )

#     detections_aftercov_class = [item[1] for item in detections_aftercov]
#     allgroundt_aftercov_class = [item[1] for item in allgroundt_aftercov]
#     detectionsgt_all_class = detections_aftercov_class + allgroundt_aftercov_class
#     detectionsgt_all_class = list(set(detectionsgt_all_class))

#     classesall = sorted(detectionsgt_all_class)
#     # print("class all redo")
#     # print(classesall)

#     # print("detection samples2")
#     # print(detections_aftercov[0])
#     # print("gt samples2")
#     # print(allgroundt_aftercov[0])

#     evaluator = Evaluator()
#     acc_AP = 0
#     validClasses = 0

#     # print("start calculate ap with boxes")
#     # detections = evaluator.PlotPrecisionRecallCurve(
#     # allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
#     # IOUThreshold=iouThreshold,  # IOU threshold
#     # method=MethodAveragePrecision.EveryPointInterpolation,
#     # showAP=True,  # Show Average Precision in the title of the plot
#     # showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
#     # savePath="/home/alex/VADE/detection_metric/Object-Detection-Metrics/results",
#     # showGraphic=False)

#     # boundingboxes=allBoundingBoxes

#     IOUThresholdmin = 0.4
#     IOUThresholdmax = 2
#     method = MethodAveragePrecision.EveryPointInterpolation

#     ret = (
#         []
#     )  # list containing metrics (precision, recall, average precision) of each class

#     # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2),annotationid])
#     groundTruths = allgroundt_aftercov
#     # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2),annotationid])
#     detections = detections_aftercov
#     # Get all classes
#     classes = classesall
#     # Loop through all bounding boxes and separate them into GTs and detections

#     # print("sample groundTruths")
#     # print(len(groundTruths))
#     # print(groundTruths[0])

#     # print("sample detections")
#     # print(len(detections))
#     # print(detections[0])

#     # print("all class")
#     # print(classes)

#     # print("start to calculate all metic for all classes")
#     for c in classes:
#         # print("class")
#         # print(c)
#         # Get only detection of class c
#         dects = []
#         [dects.append(d) for d in detections if d[1] == c]
#         # Get only ground truths of class c, use filename as key
#         gts = {}
#         npos = 0
#         for g in groundTruths:
#             if g[1] == c:
#                 npos += 1
#                 gts[g[0]] = gts.get(g[0], []) + [g]

#         # sort detections by decreasing confidence
#         dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
#         TP = np.zeros(len(dects))
#         TP_annoid = np.zeros(len(dects))
#         FP = np.zeros(len(dects))
#         # create dictionary with amount of gts for each image
#         det = {key: np.zeros(len(gts[key])) for key in gts}

#         # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
#         # print(dects)
#         # Loop through detections
#         for d in range(len(dects)):
#             # print('dect %s => %s' % (dects[d][0], dects[d][3],))
#             # Find ground truth image
#             gt = gts[dects[d][0]] if dects[d][0] in gts else []
#             iouMax = sys.float_info.min
#             for j in range(len(gt)):
#                 # print('Ground truth gt => %s' % (gt[j][3],))
#                 # print(gt[j])
#                 iou = Evaluator.iou(dects[d][3], gt[j][3])
#                 if iou > iouMax:
#                     iouMax = iou
#                     jmax = j
#             # Assign detection as true positive/don't care/false positive
#             if iouMax >= IOUThresholdmin and iouMax <= IOUThresholdmax:
#                 if det[dects[d][0]][jmax] == 0:
#                     TP[d] = 1  # count as true positive in this detection
#                     det[dects[d][0]][jmax] = 1  # flag as already 'seen'
#                     TP_annoid[d] = gt[jmax][
#                         -1
#                     ]  # sotrt related annotation ID from annotation
#                     # print("TP")
#                 else:
#                     FP[d] = 1  # count as false positive
#                     # print("FP")
#             # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
#             else:
#                 FP[d] = 1  # count as false positive
#                 # print("FP")
#         # compute precision, recall and average precision
#         acc_FP = np.cumsum(FP)
#         acc_TP = np.cumsum(TP)

#         # take all TP in the class if detection, with annoatation id
#         dects_tp_fromidx = []
#         for index in range(len(TP)):
#             if TP[index] == 1:
#                 temp = dects[index]
#                 temp.append(
#                     str(int(TP_annoid[index]))
#                 )  # add annotationID in this TP detection
#                 dects_tp_fromidx.append(temp)
#                 # dects_tp_fromidx.append(dects[index])

#         # take all FP in the class if detection, with annoatation id
#         dects_FP_fromidx = []
#         for index in range(len(FP)):
#             if FP[index] == 1:
#                 dects_FP_fromidx.append(dects[index])

#         # select TP an FN from ground truth per class
#         from_gt_TP = []
#         from_gt_FN = []
#         for gtkey in gts:
#             # print(gtkey)
#             gt_perkey = gts[gtkey]
#             gt_result = det[gtkey]
#             # print(gt_perkey)
#             # print(gt_result)
#             for gt_img_idx in range(len(gt_result)):
#                 if gt_result[gt_img_idx] == 1:
#                     from_gt_TP.append(gt_perkey[gt_img_idx])
#                 elif gt_result[gt_img_idx] == 0:
#                     from_gt_FN.append(gt_perkey[gt_img_idx])

#                     # print("detection TP, FP")
#         # print(len(dects_tp_fromidx))
#         # print(len(dects_FP_fromidx))

#         # print("gt TP, FP")
#         # print(len(from_gt_TP))
#         # print(len(from_gt_FN))

#         rec = acc_TP / npos
#         prec = np.divide(acc_TP, (acc_FP + acc_TP))
#         # Depending on the method, call the right implementation
#         if method == MethodAveragePrecision.EveryPointInterpolation:
#             [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
#         else:
#             [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)
#         # add class result in the dictionary to be returned
#         r = {
#             "class": c,
#             "precision": prec,
#             "recall": rec,
#             "AP": ap,
#             "interpolated precision": mpre,
#             "interpolated recall": mrec,
#             "total positives": npos,
#             "total TP": np.sum(TP),
#             "total FP": np.sum(FP),
#             "total TP from gt": len(from_gt_TP),
#             "total FN from gt": len(from_gt_FN),
#             "all detail TP from detection": dects_tp_fromidx,
#             "all detail FP from detection": dects_FP_fromidx,
#             "all detail TP from ground truth": from_gt_TP,
#             "all detail FN from ground truth": from_gt_FN,
#         }
#         ret.append(r)
#     # as name all detail TP from detection, all detail FP from detection, all detail TP from ground truth, all detail FN from ground truth
#     # are sore here per box per line in format [imageName,class,confidence, (bb coordinates),annotationid]

#     allPv = []
#     allRv = []
#     allAPv = []

#     for item in ret:
#         # if math.isnan(item["AP"]):
#         #    print("class")
#         #    print(item["class"])
#         #    print("ap is None")
#         #    continue
#         # else:
#         if opt.dummy_plot:
#             print("class")
#             print(item["class"])
#             print("TP")
#             print(item["total TP"])
#             print("FP")
#             print(item["total FP"])
#             print("FN")
#             print(item["total FN from gt"])
#             print("ground true positive samples")
#             print(item["total positives"])
#             print("precision")
#         # precision = item["total TP"]/(item["total TP"]+item["total FP"])
#         try:
#             precision = len(item["all detail TP from detection"]) / (
#                 len(item["all detail TP from detection"])
#                 + len(item["all detail FP from detection"])
#             )
#         except:
#             precision = 0
#         allPv.append(precision)
#         if opt.dummy_plot:
#             print(precision)
#             print("Recall")
#         # recall=item["total TP"]/item["total positives"]
#         try:
#             recall = len(item["all detail TP from ground truth"]) / (
#                 len(item["all detail TP from ground truth"])
#                 + len(item["all detail FN from ground truth"])
#             )
#         except:
#             recall = 0
#         allRv.append(recall)
#         if opt.dummy_plot:
#             print(recall)
#             print("ap")
#             print(item["AP"])
#         allAPv.append(item["AP"])

#         # print("sample 10 detail box infor in detection and ground truth")
#         # print("sample 10 all detail TP from detection")
#         # print(item["all detail TP from detection"][:10])
#         # print("sample 10 all detail FP from detection")
#         # print(item["all detail FP from detection"][:10])
#         # print("sample 10 all detail TP from ground truth")
#         # print(item["all detail TP from ground truth"][:10])
#         # print("sample 10 all detail FN from ground truth")
#         # print(item["all detail FN from ground truth"][:10])

#     # mean_presion=statistics.mean(allPv)
#     # print("mean_presion")
#     # print(mean_presion)
#     # mean_recall=statistics.mean(allRv)
#     # print("mean_recall")
#     # print(mean_recall)
#     allAPv = [float(item) for item in allAPv]
#     mean_ap = statistics.mean(allAPv)
#     if opt.dummy_plot:
#         print("mean_ap")
#         print(mean_ap)

#     return totalloss, mean_ap


# def evaluate_lmdb_outyaml_v3(
#     opt, model, test_loader, epoch, writer, encoder, nms_threshold, dboxes_default
# ):
#     yaml_out_temp1 = {"info": {"dataset_id": "111ageval"}, "output_images": []}

#     if opt.dummy_val:
#         all_det_result = glob(
#             "/home/alex/vade/map_simple/mAP/input/detection-results/*.txt"
#         )
#         all_gt_result = glob("/home/alex/vade/map_simple/mAP/input/ground-truth/*.txt")

#         all_det_result_matrix = []
#         for lines in all_det_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_det_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[2]),
#                         float(indiline_split[3]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         float(indiline_split[5]) - float(indiline_split[3]),
#                         indiline_split[1],
#                         indiline_split[0],
#                     ]
#                 )
#         all_groundt_result_matrix = []
#         for lines in all_gt_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_groundt_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[1]),
#                         float(indiline_split[2]),
#                         float(indiline_split[3]) - float(indiline_split[1]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         1,
#                         indiline_split[0],
#                     ]
#                 )
#         caculate_mAP_local(all_det_result_matrix, all_groundt_result_matrix)

#     model.eval()
#     detections = []
#     allgroundt = []
#     # category_ids = test_loader.dataset.coco.getCatIds()
#     # for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

#     # print("hello")

#     # for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):

#     dump_yaml_temp = []
#     with open(opt.yaml_out_path, "w+") as f:
#         f.write(yaml.dump(yaml_out_temp1))
#         f.write("output:" + "\n")

#         for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
#             img = img.float() / 255.0
#             # print("nbatch")
#             # print(nbatch)
#             # start = time.time()
#             # sample only for fast eval
#             # if nbatch>1:
#             #    break
#             # print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#             # print("batch")
#             # print
#             # print("shape")
#             # print(shapes)
#             # allgot_ccoform=[]

#             # print("time in one gt loop")
#             # print(time.time()-start)
#             # start = time.time()
#             # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#             # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#             # result = torch.cat(outputs, dim=1)
#             # boxinallbatchresult = torch.tensor(boxinallbatch)
#             # labelinallbatchresult = torch.tensor(labelinallbatch)

#             # print("all tensor shape overall")
#             # print(allgroundt)
#             # print(labelinallbatchresult.shape)

#             if torch.cuda.is_available():
#                 img = img.cuda()
#             with torch.no_grad():
#                 # Get predictions
#                 plabel, ploc = model(img.to(torch.float))
#                 ploc, plabel = ploc.float(), plabel.float()

#                 ploc = ploc.view(opt.batch_size, -1, 3000)
#                 plabel = plabel.view(opt.batch_size, -1, 3000)
#                 # print("predict label")
#                 # print(plabel)
#                 detection_target_inbatch = []
#                 for idx in range(ploc.shape[0]):
#                     ploc_i = ploc[idx, :, :].unsqueeze(0)
#                     plabel_i = plabel[idx, :, :].unsqueeze(0)
#                     try:
#                         # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                         result = encoder.decode_batch(
#                             ploc_i, plabel_i, nms_threshold, 200
#                         )[0]
#                     except:
#                         print("No object detected in idx: {}".format(idx))
#                         continue
#                     width = shapes[idx][0][1]
#                     height = shapes[idx][0][0]
#                     # height, width = img_size[idx]
#                     loc, label, prob = [r.cpu().numpy() for r in result]
#                     annidx = 0
#                     for loc_, label_, prob_ in zip(loc, label, prob):
#                         annidx += 1
#                         detections.append(
#                             [
#                                 paths[idx],
#                                 loc_[0] * width,
#                                 loc_[1] * height,
#                                 (loc_[2] - loc_[0]) * width,
#                                 (loc_[3] - loc_[1]) * height,
#                                 prob_,
#                                 label_,
#                             ]
#                         )
#                         # category_ids[label_ - 1]])

#                         conf = str(prob_)
#                         if float(conf) > 0.05:
#                             # conf = np.int(boxs[6])
#                             gt = np.int(label_) - 1
#                             temp_a = {
#                                 "input_id": str(paths[idx]),
#                                 "bbox": [
#                                     float(loc_[0]),
#                                     float(loc_[1]),
#                                     float(loc_[2] - loc_[0]),
#                                     float(loc_[3] - loc_[1]),
#                                 ],
#                                 # "bbox": [str((boxs[2]-boxs[4]/2)),str((boxs[3]-boxs[5]/2)),str(boxs[4]),str(boxs[5])],
#                                 # "bbox": [str((boxs[2]-boxs[4]/2)*ori_w),str((boxs[3]-boxs[5]/2)*ori_h),str(boxs[4]*ori_w),str(boxs[5]*ori_h)],
#                                 # "bbox": [str(boxs[2]),str(boxs[3]),str(boxs[4]),str(boxs[5])],
#                                 "segmentation": [],
#                                 "points": [],
#                                 "category_id": [opt.category_dic[int(gt)]],
#                                 # "category_id": [str(gt)],
#                                 "category_confidence_value": [float(conf)],
#                                 # "category_confidence_value":[str(conf)],
#                                 "label": [],
#                                 "label_conf_value": [],
#                                 "output_image_id": "",
#                                 "output_id": "1" + str(annidx) + str(paths[idx]),
#                             }
#                             dump_yaml_temp.append(temp_a)

#                         if prob_ > 0.3:
#                             detection_target_inbatch.append(
#                                 [
#                                     idx,
#                                     label_,
#                                     (loc_[0] + loc_[2]) / 2,
#                                     (loc_[3] + loc_[1]) / 2,
#                                     (loc_[2] - loc_[0]),
#                                     (loc_[3] - loc_[1]),
#                                 ]
#                             )

#                     store_bef_write = yaml.dump(dump_yaml_temp)
#                     # f.write(store_bef_write)
#                     # print("sucess dump")
#                     # print(type(store_bef_write))
#                     # dump_yaml_temp=[]

#                     if len(dump_yaml_temp) == 0:
#                         dump_yaml_temp = []
#                     else:
#                         # print(batch_i)
#                         f.write(store_bef_write)
#                         dump_yaml_temp = []

#                     # print(idx)
#                     # try:
#                     #    store_bef_write=yaml.dump(dump_yaml_temp)
#                     #    f.write(store_bef_write)
#                     # print("sucess dump")
#                     # print(type(store_bef_write))
#                     #    dump_yaml_temp=[]
#                     # except:
#                     #    print("wrong dump")
#                     #    print(type(store_bef_write))
#                     #    print(yaml.dump(list(dump_yaml_temp)))
#                     # print(dump_yaml_temp)
#                     #    print(type(dump_yaml_temp))
#                     #    print(len(dump_yaml_temp))
#                     # print(dump_yaml_temp[:5])

#             # print("time in one nms loop")
#             # print(time.time()-start)


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

        for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
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
                        # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                        result = encoder.decode_batch(
                            ploc_i, plabel_i, nms_threshold, 200
                        )[0]
                    except:
                        print("No object detected in idx: {}".format(idx))
                        continue
                    width = shapes[idx][0][1]
                    height = shapes[idx][0][0]
                    # height, width = img_size[idx]
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
                            # conf = np.int(boxs[6])
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
                        dump_yaml_temp = []
                    else:
                        # print(batch_i)
                        f.write(store_bef_write)
                        dump_yaml_temp = []


# def evaluate_lmdb_outyaml_v2(
#     opt, model, test_loader, epoch, writer, encoder, nms_threshold, dboxes_default
# ):
#     yaml_out_temp1 = {"info": {"dataset_id": "111ageval"}, "output_images": []}

#     if opt.dummy_val:
#         all_det_result = glob(
#             "/home/alex/vade/map_simple/mAP/input/detection-results/*.txt"
#         )
#         all_gt_result = glob("/home/alex/vade/map_simple/mAP/input/ground-truth/*.txt")

#         all_det_result_matrix = []
#         for lines in all_det_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_det_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[2]),
#                         float(indiline_split[3]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         float(indiline_split[5]) - float(indiline_split[3]),
#                         indiline_split[1],
#                         indiline_split[0],
#                     ]
#                 )
#         all_groundt_result_matrix = []
#         for lines in all_gt_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_groundt_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[1]),
#                         float(indiline_split[2]),
#                         float(indiline_split[3]) - float(indiline_split[1]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         1,
#                         indiline_split[0],
#                     ]
#                 )
#         caculate_mAP_local(all_det_result_matrix, all_groundt_result_matrix)

#     model.eval()
#     detections = []
#     allgroundt = []
#     # category_ids = test_loader.dataset.coco.getCatIds()
#     # for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

#     # print("hello")

#     # for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):

#     dump_yaml_temp = []
#     with open(opt.yaml_out_path, "w+") as f:
#         f.write(yaml.dump(yaml_out_temp1))
#         f.write("output:" + "\n")

#         for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
#             img = img.float() / 255.0
#             # print("nbatch")
#             # print(nbatch)
#             # start = time.time()
#             # sample only for fast eval
#             # if nbatch>1:
#             #    break
#             # print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#             # print("batch")
#             # print
#             # print("shape")
#             # print(shapes)
#             # allgot_ccoform=[]

#             # print("time in one gt loop")
#             # print(time.time()-start)
#             # start = time.time()
#             # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#             # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#             # result = torch.cat(outputs, dim=1)
#             # boxinallbatchresult = torch.tensor(boxinallbatch)
#             # labelinallbatchresult = torch.tensor(labelinallbatch)

#             # print("all tensor shape overall")
#             # print(allgroundt)
#             # print(labelinallbatchresult.shape)

#             if torch.cuda.is_available():
#                 img = img.cuda()
#             with torch.no_grad():
#                 # Get predictions
#                 ploc, plabel = model(img.to(torch.float))
#                 ploc, plabel = ploc.float(), plabel.float()
#                 # print("predict label")
#                 # print(plabel)
#                 detection_target_inbatch = []
#                 for idx in range(ploc.shape[0]):
#                     ploc_i = ploc[idx, :, :].unsqueeze(0)
#                     plabel_i = plabel[idx, :, :].unsqueeze(0)
#                     try:
#                         # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                         result = encoder.decode_batch(
#                             ploc_i, plabel_i, nms_threshold, 200
#                         )[0]
#                     except:
#                         print("No object detected in idx: {}".format(idx))
#                         continue
#                     width = shapes[idx][0][1]
#                     height = shapes[idx][0][0]
#                     # height, width = img_size[idx]
#                     loc, label, prob = [r.cpu().numpy() for r in result]
#                     annidx = 0
#                     for loc_, label_, prob_ in zip(loc, label, prob):
#                         annidx += 1
#                         detections.append(
#                             [
#                                 paths[idx],
#                                 loc_[0] * width,
#                                 loc_[1] * height,
#                                 (loc_[2] - loc_[0]) * width,
#                                 (loc_[3] - loc_[1]) * height,
#                                 prob_,
#                                 label_,
#                             ]
#                         )
#                         # category_ids[label_ - 1]])

#                         conf = str(prob_)
#                         if float(conf) > 0.05:
#                             # conf = np.int(boxs[6])
#                             gt = np.int(label_) - 1
#                             temp_a = {
#                                 "input_id": str(paths[idx]),
#                                 "bbox": [
#                                     float(loc_[0]),
#                                     float(loc_[1]),
#                                     float(loc_[2] - loc_[0]),
#                                     float(loc_[3] - loc_[1]),
#                                 ],
#                                 # "bbox": [str((boxs[2]-boxs[4]/2)),str((boxs[3]-boxs[5]/2)),str(boxs[4]),str(boxs[5])],
#                                 # "bbox": [str((boxs[2]-boxs[4]/2)*ori_w),str((boxs[3]-boxs[5]/2)*ori_h),str(boxs[4]*ori_w),str(boxs[5]*ori_h)],
#                                 # "bbox": [str(boxs[2]),str(boxs[3]),str(boxs[4]),str(boxs[5])],
#                                 "segmentation": [],
#                                 "points": [],
#                                 # opt.categpry_dic.index(int(item["category_id"][0]))
#                                 "category_id": [opt.categpry_dic[int(gt)]],
#                                 # "category_id": [str(gt)],
#                                 "category_confidence_value": [float(conf)],
#                                 # "category_confidence_value":[str(conf)],
#                                 "label": [],
#                                 "label_conf_value": [],
#                                 "output_image_id": "",
#                                 "output_id": "1" + str(annidx) + str(paths[idx]),
#                             }
#                             dump_yaml_temp.append(temp_a)

#                         if prob_ > 0.3:
#                             detection_target_inbatch.append(
#                                 [
#                                     idx,
#                                     label_,
#                                     (loc_[0] + loc_[2]) / 2,
#                                     (loc_[3] + loc_[1]) / 2,
#                                     (loc_[2] - loc_[0]),
#                                     (loc_[3] - loc_[1]),
#                                 ]
#                             )

#                     store_bef_write = yaml.dump(dump_yaml_temp)
#                     f.write(store_bef_write)
#                     # print("sucess dump")
#                     # print(type(store_bef_write))
#                     dump_yaml_temp = []

#                     # print(idx)
#                     # try:
#                     #    store_bef_write=yaml.dump(dump_yaml_temp)
#                     #    f.write(store_bef_write)
#                     # print("sucess dump")
#                     # print(type(store_bef_write))
#                     #    dump_yaml_temp=[]
#                     # except:
#                     #    print("wrong dump")
#                     #    print(type(store_bef_write))
#                     #    print(yaml.dump(list(dump_yaml_temp)))
#                     # print(dump_yaml_temp)
#                     #    print(type(dump_yaml_temp))
#                     #    print(len(dump_yaml_temp))
#                     # print(dump_yaml_temp[:5])

#             # print("time in one nms loop")
#             # print(time.time()-start)


# def evaluate_lmdb_returnloss(
#     opt,
#     model,
#     test_loader,
#     epoch,
#     writer,
#     encoder,
#     nms_threshold,
#     criterion,
#     dboxes_default,
# ):
#     model.eval()
#     detections = []
#     allgroundt = []
#     # category_ids = test_loader.dataset.coco.getCatIds()
#     # for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

#     # print("hello")
#     totalloss = 0
#     # for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
#     for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
#         img = img.float() / 255.0
#         # if nbatch>2:
#         #    break
#         start = time.time()
#         # sample only for fast eval
#         # if nbatch>1:
#         #    break
#         # print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#         # print("batch")
#         # print
#         # print("shape")
#         # print(shapes)
#         # allgot_ccoform=[]
#         # print("all tensor shape overall")
#         # print(boxinallbatchresult.shape)
#         # print(labelinallbatchresult.shape)

#         # print("img and label size")
#         # print(img.shape)
#         # print(gloc.shape)
#         # print(glabel.shape)

#         boxinallbatch = []
#         labelinallbatch = []
#         for batchidx in range(opt.batch_size):
#             # print("image shape")
#             # print(shapes[batchidx])
#             # print("new batch in xmid ymid w h")
#             tempb = targets[targets[:, 0] == batchidx]
#             # print(tempb)
#             temploc = tempb[:, 2:]

#             templabel = tempb[:, 1] + 1
#             # print(temploc)
#             # print(templabel)

#             # temploc_updateform2=[[item[2]-item[4]/2,item[3]-item[5]/2,item[4],item[5],item[1]] for item in zip(tempb,)]
#             temploc_updateform2 = [
#                 [
#                     item[0][0] - item[0][2] / 2,
#                     item[0][1] - item[0][3] / 2,
#                     item[0][2],
#                     item[0][3],
#                     item[1],
#                 ]
#                 for item in list(zip(temploc, templabel))
#             ]
#             templocredotensor2 = torch.as_tensor(temploc_updateform2)
#             # print("templocredotensor2 in x1 y1 w h")
#             # print(templocredotensor2)

#             temploc_updateform = [
#                 [
#                     item[0] - item[2] / 2,
#                     item[1] - item[3] / 2,
#                     item[0] + item[2] / 2,
#                     item[1] + item[3] / 2,
#                 ]
#                 for item in temploc
#             ]
#             templocredotensor = torch.as_tensor(temploc_updateform)
#             # print(templocredotensor)
#             # print("default box shape")
#             # print(dboxes_default.shape)
#             bboxes, labels = encode_local(
#                 bboxes_in=templocredotensor,
#                 labels_in=templabel.to(torch.long),
#                 dboxes_default_encode=dboxes_default,
#             )
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             boxinallbatch.append(bboxes.numpy())
#             labelinallbatch.append(labels.numpy())

#             # temploc_updateform=[[item[0]-item[2]/2,item[1]-item[3]/2,item[2],item[3]] for item in temploc]
#             ##templocredotensor = torch.as_tensor(temploc_updateform)
#             # print("templocredotensor")
#             # print(templocredotensor)
#             # bboxes, labels = encode_local(bboxes_in=templocredotensor, labels_in=templabel.to(torch.long))
#             # print(bboxes.shape)
#             # print(labels.shape)
#             # print(bboxes[:10])
#             # print(labels[labels!=0])
#             # boxinallbatch.append(bboxes.numpy())
#             # labelinallbatch.append(labels.numpy())
#             width = shapes[batchidx][0][1]
#             height = shapes[batchidx][0][0]
#             for value in templocredotensor2:
#                 allgroundt.append(
#                     [
#                         paths[batchidx],
#                         value[0] * width,
#                         value[1] * height,
#                         value[2] * width,
#                         value[3] * height,
#                         1,
#                         value[4],
#                     ]
#                 )

#         gloc = torch.tensor(boxinallbatch)
#         glabel = torch.tensor(labelinallbatch)
#         # print("time in one gt loop")
#         # print(time.time()-start)
#         start = time.time()
#         # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#         # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#         # result = torch.cat(outputs, dim=1)
#         # boxinallbatchresult = torch.tensor(boxinallbatch)
#         # labelinallbatchresult = torch.tensor(labelinallbatch)

#         # print("all tensor shape overall")
#         # print(allgroundt)
#         # print(labelinallbatchresult.shape)

#         if torch.cuda.is_available():
#             img = img.cuda()
#             gloc = gloc.cuda()
#             glabel = glabel.cuda()

#         # ploc, plabel = model(img.to(torch.float))
#         # ploc, plabel = ploc.float(), plabel.float()

#         with torch.no_grad():
#             # Get predictions
#             ploc, plabel = model(img.to(torch.float))
#             ploc, plabel = ploc.float(), plabel.float()

#             gloc = gloc.transpose(1, 2).contiguous()
#             loss = criterion(ploc.cpu(), plabel.cpu(), gloc.cpu(), glabel.cpu())
#             # print("predict label")
#             # print(plabel)
#             detection_target_inbatch = []
#             for idx in range(ploc.shape[0]):
#                 ploc_i = ploc[idx, :, :].unsqueeze(0)
#                 plabel_i = plabel[idx, :, :].unsqueeze(0)
#                 try:
#                     # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                     result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[
#                         0
#                     ]
#                 except:
#                     print("No object detected in idx: {}".format(idx))
#                     continue
#                 width = shapes[idx][0][1]
#                 height = shapes[idx][0][0]
#                 # height, width = img_size[idx]
#                 loc, label, prob = [r.cpu().numpy() for r in result]
#                 for loc_, label_, prob_ in zip(loc, label, prob):
#                     detections.append(
#                         [
#                             paths[idx],
#                             loc_[0] * width,
#                             loc_[1] * height,
#                             (loc_[2] - loc_[0]) * width,
#                             (loc_[3] - loc_[1]) * height,
#                             prob_,
#                             label_,
#                         ]
#                     )
#                     # category_ids[label_ - 1]])
#                     if prob_ > 0.3:
#                         detection_target_inbatch.append(
#                             [
#                                 idx,
#                                 label_,
#                                 (loc_[0] + loc_[2]) / 2,
#                                 (loc_[3] + loc_[1]) / 2,
#                                 (loc_[2] - loc_[0]),
#                                 (loc_[3] - loc_[1]),
#                             ]
#                         )

#         # print("time in one nms loop")
#         # print(time.time()-start)

#         # if nbatch < 50:
#         if opt.dummy_plot and nbatch < 20:
#             # if opt.dummy_plot and (epoch+1)%10==0 and nbatch < 20:
#             # print(paths)
#             f = f"test_batch{nbatch}_labels.jpg"  # filename
#             targetsny = targets.numpy()
#             targetsny[:, 1] += 1
#             plot_images(
#                 img,
#                 targetsny,
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # labels
#             f = f"test_batch{nbatch}_pred.jpg"
#             plot_images(
#                 img,
#                 torch.as_tensor(detection_target_inbatch),
#                 [str(item) for item in paths],
#                 f,
#                 [str(item) for item in range(100)],
#             )  # predictions

#     totalloss += loss
#     detections = np.array(detections, dtype=np.float32)
#     allgroundt = np.array(allgroundt, dtype=np.float32)

#     print("detect")
#     detections_class = [item[-1] for item in detections]
#     detections_class_u = list(set(detections_class))
#     print("all detect class")
#     print(detections_class_u)
#     allgroundt_class = [item[-1] for item in allgroundt]
#     allgroundt_class_u = list(set(allgroundt_class))
#     print("all gt class")
#     print(allgroundt_class_u)

#     # print(detections[0])
#     # print("allgroundt")
#     # print(allgroundt[0])

#     allclassmap = caculate_mAP_local(detections, allgroundt)

#     return totalloss, allclassmap


# def evaluate_lmdb_outyaml(
#     opt, model, test_loader, epoch, writer, encoder, nms_threshold, dboxes_default
# ):
#     yaml_out_temp1 = {"info": {"dataset_id": "111ageval"}, "output_images": []}

#     if opt.dummy_val:
#         all_det_result = glob(
#             "/home/alex/vade/map_simple/mAP/input/detection-results/*.txt"
#         )
#         all_gt_result = glob("/home/alex/vade/map_simple/mAP/input/ground-truth/*.txt")

#         all_det_result_matrix = []
#         for lines in all_det_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_det_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[2]),
#                         float(indiline_split[3]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         float(indiline_split[5]) - float(indiline_split[3]),
#                         indiline_split[1],
#                         indiline_split[0],
#                     ]
#                 )
#         all_groundt_result_matrix = []
#         for lines in all_gt_result:
#             # print(lines)
#             try:
#                 dfg = pd.read_csv(lines, engine="python", sep="/n", header=None)
#                 allscore = [x[0] for x in dfg.values]
#             except:
#                 continue
#             # print(allscore)
#             for indiline in allscore:
#                 indiline_split = indiline.split(" ")
#                 # print(indiline_split)
#                 all_groundt_result_matrix.append(
#                     [
#                         lines.split("/")[-1],
#                         float(indiline_split[1]),
#                         float(indiline_split[2]),
#                         float(indiline_split[3]) - float(indiline_split[1]),
#                         float(indiline_split[4]) - float(indiline_split[2]),
#                         1,
#                         indiline_split[0],
#                     ]
#                 )
#         caculate_mAP_local(all_det_result_matrix, all_groundt_result_matrix)

#     model.eval()
#     detections = []
#     allgroundt = []
#     # category_ids = test_loader.dataset.coco.getCatIds()
#     # for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):

#     # print("hello")

#     # for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):

#     dump_yaml_temp = []
#     with open(opt.yaml_out_path, "w+") as f:
#         f.write(yaml.dump(yaml_out_temp1))
#         f.write("output:" + "\n")

#         for nbatch, (img, targets, paths, shapes) in enumerate(test_loader):
#             img = img.float() / 255.0
#             # print("nbatch")
#             # print(nbatch)
#             # start = time.time()
#             # sample only for fast eval
#             # if nbatch>1:
#             #    break
#             # print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#             # print("batch")
#             # print
#             # print("shape")
#             # print(shapes)
#             # allgot_ccoform=[]

#             # print("time in one gt loop")
#             # print(time.time()-start)
#             # start = time.time()
#             # boxinallbatchresult = torch.cat(boxinallbatch, dim=-1)
#             # labelinallbatchresult = torch.cat(labelinallbatch, dim=-1)
#             # result = torch.cat(outputs, dim=1)
#             # boxinallbatchresult = torch.tensor(boxinallbatch)
#             # labelinallbatchresult = torch.tensor(labelinallbatch)

#             # print("all tensor shape overall")
#             # print(allgroundt)
#             # print(labelinallbatchresult.shape)

#             if torch.cuda.is_available():
#                 img = img.cuda()
#             with torch.no_grad():
#                 # Get predictions
#                 ploc, plabel = model(img.to(torch.float))
#                 ploc, plabel = ploc.float(), plabel.float()
#                 # print("predict label")
#                 # print(plabel)
#                 detection_target_inbatch = []
#                 for idx in range(ploc.shape[0]):
#                     ploc_i = ploc[idx, :, :].unsqueeze(0)
#                     plabel_i = plabel[idx, :, :].unsqueeze(0)
#                     try:
#                         # result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
#                         result = encoder.decode_batch(
#                             ploc_i, plabel_i, nms_threshold, 200
#                         )[0]
#                     except:
#                         print("No object detected in idx: {}".format(idx))
#                         continue
#                     width = shapes[idx][0][1]
#                     height = shapes[idx][0][0]
#                     # height, width = img_size[idx]
#                     loc, label, prob = [r.cpu().numpy() for r in result]
#                     annidx = 0
#                     for loc_, label_, prob_ in zip(loc, label, prob):
#                         annidx += 1
#                         detections.append(
#                             [
#                                 paths[idx],
#                                 loc_[0] * width,
#                                 loc_[1] * height,
#                                 (loc_[2] - loc_[0]) * width,
#                                 (loc_[3] - loc_[1]) * height,
#                                 prob_,
#                                 label_,
#                             ]
#                         )
#                         # category_ids[label_ - 1]])

#                         conf = str(prob_)
#                         if float(conf) > 0.05:
#                             # conf = np.int(boxs[6])
#                             gt = np.int(label_) - 1
#                             temp_a = {
#                                 "input_id": str(paths[idx]),
#                                 "bbox": [
#                                     float(loc_[0]),
#                                     float(loc_[1]),
#                                     float(loc_[2] - loc_[0]),
#                                     float(loc_[3] - loc_[1]),
#                                 ],
#                                 # "bbox": [str((boxs[2]-boxs[4]/2)),str((boxs[3]-boxs[5]/2)),str(boxs[4]),str(boxs[5])],
#                                 # "bbox": [str((boxs[2]-boxs[4]/2)*ori_w),str((boxs[3]-boxs[5]/2)*ori_h),str(boxs[4]*ori_w),str(boxs[5]*ori_h)],
#                                 # "bbox": [str(boxs[2]),str(boxs[3]),str(boxs[4]),str(boxs[5])],
#                                 "segmentation": [],
#                                 "points": [],
#                                 # opt.categpry_dic.index(int(item["category_id"][0]))
#                                 "category_id": [opt.categpry_dic[int(gt)]],
#                                 # "category_id": [str(gt)],
#                                 "category_confidence_value": [float(conf)],
#                                 # "category_confidence_value":[str(conf)],
#                                 "label": [],
#                                 "label_conf_value": [],
#                                 "output_image_id": "",
#                                 "output_id": "1" + str(annidx) + str(paths[idx]),
#                             }
#                             dump_yaml_temp.append(temp_a)

#                         if prob_ > 0.3:
#                             detection_target_inbatch.append(
#                                 [
#                                     idx,
#                                     label_,
#                                     (loc_[0] + loc_[2]) / 2,
#                                     (loc_[3] + loc_[1]) / 2,
#                                     (loc_[2] - loc_[0]),
#                                     (loc_[3] - loc_[1]),
#                                 ]
#                             )

#                     store_bef_write = yaml.dump(dump_yaml_temp)
#                     f.write(store_bef_write)
#                     # print("sucess dump")
#                     # print(type(store_bef_write))
#                     dump_yaml_temp = []

#                     # print(idx)
#                     # try:
#                     #    store_bef_write=yaml.dump(dump_yaml_temp)
#                     #    f.write(store_bef_write)
#                     # print("sucess dump")
#                     # print(type(store_bef_write))
#                     #    dump_yaml_temp=[]
#                     # except:
#                     #    print("wrong dump")
#                     #    print(type(store_bef_write))
#                     #    print(yaml.dump(list(dump_yaml_temp)))
#                     # print(dump_yaml_temp)
#                     #    print(type(dump_yaml_temp))
#                     #    print(len(dump_yaml_temp))
#                     # print(dump_yaml_temp[:5])

#             # print("time in one nms loop")
#             # print(time.time()-start)


# def evaluate(model, test_loader, epoch, writer, encoder, nms_threshold):
#     model.eval()
#     detections = []
#     category_ids = test_loader.dataset.coco.getCatIds()
#     for nbatch, (img, img_id, img_size, _, _) in enumerate(test_loader):
#         print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
#         if torch.cuda.is_available():
#             img = img.cuda()
#         with torch.no_grad():
#             # Get predictions
#             ploc, plabel = model(img)
#             ploc, plabel = ploc.float(), plabel.float()

#             for idx in range(ploc.shape[0]):
#                 ploc_i = ploc[idx, :, :].unsqueeze(0)
#                 plabel_i = plabel[idx, :, :].unsqueeze(0)
#                 try:
#                     result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[
#                         0
#                     ]
#                 except:
#                     print("No object detected in idx: {}".format(idx))
#                     continue

#                 height, width = img_size[idx]
#                 loc, label, prob = [r.cpu().numpy() for r in result]
#                 for loc_, label_, prob_ in zip(loc, label, prob):
#                     detections.append(
#                         [
#                             img_id[idx],
#                             loc_[0] * width,
#                             loc_[1] * height,
#                             (loc_[2] - loc_[0]) * width,
#                             (loc_[3] - loc_[1]) * height,
#                             prob_,
#                             category_ids[label_ - 1],
#                         ]
#                     )

#     detections = np.array(detections, dtype=np.float32)

#     coco_eval = COCOeval(
#         test_loader.dataset.coco,
#         test_loader.dataset.coco.loadRes(detections),
#         iouType="bbox",
#     )
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()

#     writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)


# def caculate_mAP_local(detection_in, gt_in):
#     MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)

#     args = SimpleNamespace()
#     # if there are no classes to ignore then replace None by empty list
#     # if args.ignore is None:
#     args.ignore = []
#     args.quiet = False
#     specific_iou_flagged = False

#     # make sure that the cwd() is the location of the python script (so that every path makes sense)
#     # os.chdir(os.path.dirname(os.path.abspath(__file__)))

#     GT_PATH = os.path.join(os.getcwd(), "input", "ground-truth")
#     DR_PATH = os.path.join(os.getcwd(), "input", "detection-results")
#     # if there are no images then no animation can be shown
#     IMG_PATH = os.path.join(os.getcwd(), "input", "images-optional")
#     if os.path.exists(IMG_PATH):
#         for dirpath, dirnames, files in os.walk(IMG_PATH):
#             if not files:
#                 # no image files found
#                 args.no_animation = True
#     else:
#         args.no_animation = True

#     # try to import OpenCV if the user didn't choose the option --no-animation
#     show_animation = False
#     args.no_animation = True
#     if not args.no_animation:
#         try:
#             import cv2

#             show_animation = True
#         except ImportError:
#             print('"opencv-python" not found, please install to visualize the results.')
#             args.no_animation = True

#     # try to import Matplotlib if the user didn't choose the option --no-plot
#     draw_plot = False
#     args.no_plot = True
#     if not args.no_plot:
#         try:
#             import matplotlib.pyplot as plt

#             draw_plot = True
#         except ImportError:
#             print(
#                 '"matplotlib" not found, please install it to get the resulting plots.'
#             )
#             args.no_plot = True

#     """
#     Draw plot using Matplotlib
#     """
#     """
#     Create a ".temp_files/" and "output/" directory
#     """
#     TEMP_FILES_PATH = ".temp_files"
#     if not os.path.exists(TEMP_FILES_PATH):  # if it doesn't exist already
#         os.makedirs(TEMP_FILES_PATH)
#     output_files_path = "outputmapcal"
#     if os.path.exists(output_files_path):  # if it exist already
#         # reset the output directory
#         shutil.rmtree(output_files_path)

#     os.makedirs(output_files_path)

#     """
#     ground-truth
#         Load each of the ground-truth files into a temporary ".json" file.
#         Create a list of all the class names present in the ground-truth (gt_classes).
#     """
#     # get a list with the ground-truth files
#     # ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
#     # if len(ground_truth_files_list) == 0:
#     #    error("Error: No ground-truth files found!")
#     # ground_truth_files_list.sort()
#     # dictionary with counter per class
#     gt_counter_per_class = {}
#     counter_images_per_class = {}

#     gt_files = []

#     # use list detection result
#     gt_imgid = [item[0] for item in gt_in]

#     gt_imgid = list(set(gt_imgid))

#     for txt_file in gt_imgid:
#         select_gt = [item for item in gt_in if item[0] == txt_file]
#         # print(txt_file)
#         file_id = txt_file
#         file_id = str(txt_file)
#         # check if there is a correspondent detection-results file
#         temp_path = os.path.join(DR_PATH, (file_id + ".txt"))

#         # lines_list = file_lines_to_list(txt_file)
#         # create ground-truth dictionary
#         bounding_boxes = []
#         is_difficult = False
#         already_seen_classes = []

#         for line in select_gt:
#             try:
#                 class_name = str(line[-1])
#                 left = line[1]
#                 top = line[2]
#                 right = line[1] + line[3]
#                 bottom = line[2] + line[4]
#                 # class_name, left, top, right, bottom = line.split()
#             except ValueError:
#                 error_msg = "Error: File " + txt_file + " in the wrong format.\n"
#                 error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
#                 error_msg += " Received: " + line
#                 error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
#                 error_msg += 'by running the script "remove_space.py" or "rename_class.py" in the "extra/" folder.'
#                 error(error_msg)
#             # check if class is in the ignore list, if yes skip
#             if class_name in args.ignore:
#                 continue
#             bbox = str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
#             # print(" bbox")
#             # print(bbox)
#             if is_difficult:
#                 bounding_boxes.append(
#                     {
#                         "class_name": class_name,
#                         "bbox": bbox,
#                         "used": False,
#                         "difficult": True,
#                     }
#                 )
#                 is_difficult = False
#             else:
#                 bounding_boxes.append(
#                     {"class_name": class_name, "bbox": bbox, "used": False}
#                 )
#                 # count that object
#                 if class_name in gt_counter_per_class:
#                     gt_counter_per_class[class_name] += 1
#                 else:
#                     # if class didn't exist yet
#                     gt_counter_per_class[class_name] = 1

#                 if class_name not in already_seen_classes:
#                     if class_name in counter_images_per_class:
#                         counter_images_per_class[class_name] += 1
#                     else:
#                         # if class didn't exist yet
#                         counter_images_per_class[class_name] = 1
#                     already_seen_classes.append(class_name)

#         # dump bounding_boxes into a ".json" file
#         new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
#         gt_files.append(new_temp_file)
#         with open(new_temp_file, "w") as outfile:
#             json.dump(bounding_boxes, outfile)

#     gt_classes = list(gt_counter_per_class.keys())
#     # let's sort the classes alphabetically
#     gt_classes = sorted(gt_classes)
#     n_classes = len(gt_classes)
#     # print(gt_classes)
#     # print(gt_counter_per_class)

#     """
#     detection-results
#         Load each of the detection-results files into a temporary ".json" file.
#     """
#     # get a list with the detection-results files
#     # dr_files_list = glob.glob(DR_PATH + '/*.txt')
#     # dr_files_list.sort()

#     # use list detection result
#     detect_in = detection_in
#     detect_imgid = [item[0] for item in detect_in]

#     dr_files_list = list(set(detect_imgid))

#     for class_index, class_name in enumerate(gt_classes):
#         bounding_boxes = []
#         for txt_file in dr_files_list:
#             # print(txt_file)
#             # the first time it checks if all the corresponding ground-truth files exist
#             file_id = str(txt_file)
#             # file_id = os.path.basename(os.path.normpath(file_id))
#             temp_path = os.path.join(GT_PATH, (str(file_id) + ".txt"))

#             # lines = file_lines_to_list(txt_file)
#             select_detect = [item for item in detect_in if item[0] == txt_file]
#             for line in select_detect:
#                 try:
#                     # tmp_class_name, confidence, left, top, right, bottom = line.split()
#                     tmp_class_name = str(line[-1])
#                     confidence = str(line[-2])
#                     left = line[1]
#                     top = line[2]
#                     right = line[1] + line[3]
#                     bottom = line[2] + line[4]
#                 except ValueError:
#                     error_msg = "Error: File " + txt_file + " in the wrong format.\n"
#                     error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
#                     error_msg += " Received: " + line
#                     error(error_msg)
#                 if tmp_class_name == class_name:
#                     # print("match")
#                     bbox = (
#                         str(left)
#                         + " "
#                         + str(top)
#                         + " "
#                         + str(right)
#                         + " "
#                         + str(bottom)
#                     )
#                     bounding_boxes.append(
#                         {"confidence": confidence, "file_id": file_id, "bbox": bbox}
#                     )
#                     # print(bounding_boxes)
#         # sort detection-results by decreasing confidence
#         bounding_boxes.sort(key=lambda x: float(x["confidence"]), reverse=True)
#         with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", "w") as outfile:
#             json.dump(bounding_boxes, outfile)

#     """
#     Calculate the AP for each class
#     """
#     sum_AP = 0.0
#     ap_dictionary = {}
#     lamr_dictionary = {}
#     # open file to store the output
#     with open(output_files_path + "/output.txt", "w") as output_file:
#         output_file.write("# AP and precision/recall per class\n")
#         count_true_positives = {}
#         for class_index, class_name in enumerate(gt_classes):
#             count_true_positives[class_name] = 0
#             """
#             Load detection-results of that class
#             """
#             dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
#             dr_data = json.load(open(dr_file))

#             """
#             Assign detection-results to ground-truth objects
#             """
#             nd = len(dr_data)
#             tp = [0] * nd  # creates an array of zeros of size nd
#             fp = [0] * nd
#             for idx, detection in enumerate(dr_data):
#                 file_id = detection["file_id"]
#                 if show_animation:
#                     # find ground truth image
#                     ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
#                     # tifCounter = len(glob.glob1(myPath,"*.tif"))
#                     if len(ground_truth_img) == 0:
#                         error("Error. Image not found with id: " + file_id)
#                     elif len(ground_truth_img) > 1:
#                         error("Error. Multiple image with id: " + file_id)
#                     else:  # found image
#                         # print(IMG_PATH + "/" + ground_truth_img[0])
#                         # Load image
#                         img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
#                         # load image with draws of multiple detections
#                         img_cumulative_path = (
#                             output_files_path + "/images/" + ground_truth_img[0]
#                         )
#                         if os.path.isfile(img_cumulative_path):
#                             img_cumulative = cv2.imread(img_cumulative_path)
#                         else:
#                             img_cumulative = img.copy()
#                         # Add bottom border to image
#                         bottom_border = 60
#                         BLACK = [0, 0, 0]
#                         img = cv2.copyMakeBorder(
#                             img,
#                             0,
#                             bottom_border,
#                             0,
#                             0,
#                             cv2.BORDER_CONSTANT,
#                             value=BLACK,
#                         )
#                 # assign detection-results to ground truth object if any
#                 # open ground-truth with that file_id
#                 gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
#                 ground_truth_data = json.load(open(gt_file))
#                 ovmax = -1
#                 gt_match = -1
#                 # load detected object bounding-box
#                 bb = [float(x) for x in detection["bbox"].split()]
#                 for obj in ground_truth_data:
#                     # look for a class_name match
#                     if obj["class_name"] == class_name:
#                         bbgt = [float(x) for x in obj["bbox"].split()]
#                         bi = [
#                             max(bb[0], bbgt[0]),
#                             max(bb[1], bbgt[1]),
#                             min(bb[2], bbgt[2]),
#                             min(bb[3], bbgt[3]),
#                         ]
#                         iw = bi[2] - bi[0] + 1
#                         ih = bi[3] - bi[1] + 1
#                         if iw > 0 and ih > 0:
#                             # compute overlap (IoU) = area of intersection / area of union
#                             ua = (
#                                 (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
#                                 + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
#                                 - iw * ih
#                             )
#                             ov = iw * ih / ua
#                             if ov > ovmax:
#                                 ovmax = ov
#                                 gt_match = obj

#                 # assign detection as true positive/don't care/false positive
#                 if show_animation:
#                     status = "NO MATCH FOUND!"  # status is only used in the animation
#                 # set minimum overlap
#                 min_overlap = MINOVERLAP
#                 if specific_iou_flagged:
#                     if class_name in specific_iou_classes:
#                         index = specific_iou_classes.index(class_name)
#                         min_overlap = float(iou_list[index])
#                 if ovmax >= min_overlap:
#                     if "difficult" not in gt_match:
#                         if not bool(gt_match["used"]):
#                             # true positive
#                             tp[idx] = 1
#                             gt_match["used"] = True
#                             count_true_positives[class_name] += 1
#                             # update the ".json" file
#                             with open(gt_file, "w") as f:
#                                 f.write(json.dumps(ground_truth_data))
#                             if show_animation:
#                                 status = "MATCH!"
#                         else:
#                             # false positive (multiple detection)
#                             fp[idx] = 1
#                             if show_animation:
#                                 status = "REPEATED MATCH!"
#                 else:
#                     # false positive
#                     fp[idx] = 1
#                     if ovmax > 0:
#                         status = "INSUFFICIENT OVERLAP"

#                 """
#                 Draw image to show animation
#                 """

#             # print(tp)
#             # compute precision/recall
#             cumsum = 0
#             for idx, val in enumerate(fp):
#                 fp[idx] += cumsum
#                 cumsum += val
#             cumsum = 0
#             for idx, val in enumerate(tp):
#                 tp[idx] += cumsum
#                 cumsum += val
#             # print(tp)
#             rec = tp[:]
#             for idx, val in enumerate(tp):
#                 rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
#             # print(rec)
#             prec = tp[:]
#             for idx, val in enumerate(tp):
#                 prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
#             # print(prec)

#             ap, mrec, mprec = voc_ap(rec[:], prec[:])
#             sum_AP += ap
#             text = (
#                 "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
#             )  # class_name + " AP = {0:.2f}%".format(ap*100)
#             print("class")
#             print(class_name)
#             print("AP")
#             print(ap)
#             """
#             Write to output.txt
#             """
#             rounded_prec = ["%.2f" % elem for elem in prec]
#             rounded_rec = ["%.2f" % elem for elem in rec]
#             output_file.write(
#                 text
#                 + "\n Precision: "
#                 + str(rounded_prec)
#                 + "\n Recall :"
#                 + str(rounded_rec)
#                 + "\n\n"
#             )
#             if not args.quiet:
#                 print(text)
#             ap_dictionary[class_name] = ap

#             n_images = counter_images_per_class[class_name]
#             lamr, mr, fppi = log_average_miss_rate(
#                 np.array(prec), np.array(rec), n_images
#             )
#             lamr_dictionary[class_name] = lamr

#             """
#             Draw plot
#             """
#         output_file.write("\n# mAP of all classes\n")
#         mAP = sum_AP / n_classes
#         allclassmap = mAP
#         text = "mAP = {0:.2f}%".format(mAP * 100)
#         output_file.write(text + "\n")
#         print(text)

#     """
#     Draw false negatives
#     """

#     # remove the temp_files directory
#     shutil.rmtree(TEMP_FILES_PATH)

#     """
#     Count total of detection-results
#     """
#     # iterate through all the files
#     det_counter_per_class = {}
#     for txt_file in dr_files_list:
#         select_detection = [item for item in detect_in if item[0] == txt_file]
#         for line in select_detection:
#             class_name = line[-1]
#             # check if class is in the ignore list, if yes skip
#             if class_name in args.ignore:
#                 continue
#             # count that object
#             if class_name in det_counter_per_class:
#                 det_counter_per_class[class_name] += 1
#             else:
#                 # if class didn't exist yet
#                 det_counter_per_class[class_name] = 1
#     # print(det_counter_per_class)
#     dr_classes = list(det_counter_per_class.keys())

#     """
#     Finish counting true positives
#     """
#     for class_name in dr_classes:
#         # if class exists in detection-result but not in ground-truth then there are no true positives in that class
#         if class_name not in gt_classes:
#             count_true_positives[class_name] = 0
#     # print(count_true_positives)
#     return allclassmap
