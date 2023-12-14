import random
import ruamel.yaml
import torch.utils.data
import yaml
from tqdm import tqdm

# dataloader
# from vitg.utils.datasets import create_dataloader


yamldy = ruamel.yaml.YAML()

import argparse
import math
import os
import pickle
import time
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import Dataset


def select_device(device="", batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert (
            torch.cuda.is_available()
        ), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024**2  # bytes to MB
        ng = torch.cuda.device_count()
        if (
            ng > 1 and batch_size
        ):  # check that batch_size is compatible with device_count
            assert (
                batch_size % ng == 0
            ), "batch-size %g not multiple of GPU count %g" % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA "
        for i in range(ng):
            if i == 1:
                s = " " * len(s)
            print(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                % (s, i, x[i].name, x[i].total_memory / c)
            )
    else:
        print("Using CPU")

    print("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def unpack_img_lmdb(buf, iscolor=1):
    img = np.frombuffer(buf, dtype=np.uint8)
    assert cv2 is not None
    img = cv2.imdecode(img, iscolor)
    return img


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    # img = self.imgs[index]

    img = unpack_img_lmdb(self.unpacked[0])

    # img = self.unpack_img(self.samples[index][0])

    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    # if img is None:  # not cached
    # path = self.img_files[index]
    # img = cv2.imread(path)  # BGR
    #    img = self.unpack_img(sample[0])
    #    assert img is not None, 'Image Not Found ' + path
    #    h0, w0 = img.shape[:2]  # orig hw
    #    r = self.img_size / max(h0, w0)  # resize image to img_size
    #    if r != 1:  # always resize down, only resize up if training with augmentation
    #        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
    #        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    #    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    # else:
    #    return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        opt=None,
    ):
        # try:
        #    f = []  # image files
        #    for p in path if isinstance(path, list) else [path]:
        #        p = str(Path(p))  # os-agnostic
        #        parent = str(Path(p).parent) + os.sep
        #        if os.path.isfile(p):  # file
        #            with open(p, 'r') as t:
        #                t = t.read().splitlines()
        #                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        #        elif os.path.isdir(p):  # folder
        #            f += glob.iglob(p + os.sep + '*.*')
        #        else:
        #            raise Exception('%s does not exist' % p)
        #    self.img_files = sorted(
        #        [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats])
        # except Exception as e:
        #    raise Exception('Error loading data from %s: %s\nSee %s' % (path, e, help_url))
        # print("caty ids")
        # print(opt.categpry_dic)
        # self.categpry_dic = opt.categpry_dic
        # print("self caty ids")
        # print(self.categpry_dic)
        self.env = lmdb.open(
            path,
            subdir=os.path.isdir(path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))
        self.unpacked = []

        # self.samples = np.load(path, allow_pickle=True, encoding='latin1')
        n = self.length
        print("loaded all samples")
        print(path)
        print(self.length)
        assert n > 0, f"No images found in {path}. See {help_url}"
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images

        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = (
            self.augment and not self.rect
        )  # load 4 images at a time into a mosaic (only during training)
        # self.mosaic = False
        # print("not use mosaic")
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride

        # Define labels
        # self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in
        #                    self.img_files]
        # self.label_files = ["/home/alex/data/all_label_txt_coco_abod/" + x.split("/")[-1].replace(".jpg", ".txt").replace(".png", ".txt")
        # for x in self.img_files]

        # Check cache
        # cache_path = str(Path(os.getcwd()) + '.cache'  # cached labels
        # print(cache_path)
        # if os.path.isfile(cache_path):
        #    cache = torch.load(cache_path)  # load
        #    if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
        #        cache = self.cache_labels(cache_path)  # re-cache
        # else:
        #    cache = self.cache_labels(cache_path)  # cache
        # cache = self.cache_labels(cache_path)  # cache all ways first to avoid error

        # Get labels
        # labels, shapes = zip(*[cache[x] for x in self.img_files])
        # self.shapes = np.array(shapes, dtype=np.float64)
        # self.labels = list(labels)

        # satasample_img = [unpack_img2(item[0],iscolor=1) for item in self.samples]
        # satasample_img_shape = [[item.shape[1], item.shape[0]] for item in satasample_img]
        # self.shapes = np.array(satasample_img_shape)

        shapes_off = []
        for index in range(self.length):
            with self.env.begin(write=False) as txn:
                byteflow_shape = txn.get(self.keys[index])
            unpacked_shape = pickle.loads(byteflow_shape)
            shapes_off.append([unpacked_shape[3], unpacked_shape[4]])
        self.shapes = np.array(shapes_off)

        # print("get all shape")
        # print(len(self.shapes))

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  #
            # print(s)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            # self.img_files = [self.img_files[i] for i in irect]
            # self.label_files = [self.label_files[i] for i in irect]
            # self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int)
                * stride
            )

    def __len__(self):
        return self.length

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self
    def unpack_img(self, buf):
        assert cv2 is not None
        return cv2.imdecode(buf, 1)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        self.unpacked = pickle.loads(byteflow)

        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp["mixup"]:
                img2, labels2 = load_mosaic(
                    self, random.randint(0, len(self.samples) - 1)
                )
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            # print("load img without mosaic")
            # print(index.shape)
            # print(index)
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = (
                self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            )  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            # x = np.array(self.samples[index][1], dtype=np.float)
            # print("unpack label")
            # print(self.categpry_dic.index(item["category_id"][0]))
            # self.categpry_dic
            unpacked_labels_nomosaic = [
                [
                    item["category_id"][0],
                    item["bbox"][0],
                    item["bbox"][1],
                    item["bbox"][2],
                    item["bbox"][3],
                ]
                for item in self.unpacked[1]
            ]
            # unpacked_labels_nomosaic=[[self.categpry_dic.index(int(item["category_id"][0])),item["bbox"][0],item["bbox"][1],item["bbox"][2],item["bbox"][3]] for item in self.unpacked[1]]
            x = np.array(unpacked_labels_nomosaic, dtype=np.float)

            # print("labels")
            # print(x)
            # x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = (
                    ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                )  # pad width
                labels[:, 2] = (
                    ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                )  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

            # Augment colorspace
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # print("lables out")
        # print(labels_out)
        # return torch.from_numpy(img), labels_out, self.img_files[index], shapes
        return torch.from_numpy(img), labels_out, self.unpacked[5], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    opt,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    local_rank=-1,
    world_size=1,
):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    # with torch_distributed_zero_first(local_rank):
    dataset = LoadImagesAndLabels(
        path,
        imgsz,
        batch_size,
        augment=augment,  # augment images
        hyp=hyp,  # augmentation hyperparameters
        rect=rect,  # rectangular training
        cache_images=cache,
        single_cls=opt.single_cls,
        stride=int(stride),
        pad=pad
        # opt=opt
    )

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset)
        if local_rank != -1
        else None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    return dataloader, dataset


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, nc, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)

        self.module_defs[144]["filters"] = (nc + 5) * 3
        self.module_defs[145]["classes"] = nc
        self.module_defs[159]["filters"] = (nc + 5) * 3
        self.module_defs[160]["classes"] = nc
        self.module_defs[174]["filters"] = (nc + 5) * 3
        self.module_defs[175]["classes"] = nc
        # print(self.module_defs[144])
        # print(self.module_defs[145])
        # print(self.module_defs[159])
        # print(self.module_defs[160])
        # print(self.module_defs[174])
        # print(self.module_defs[175])

        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array(
            [0, 2, 5], dtype=np.int32
        )  # (int32) version info: major, minor, revision
        self.seen = np.array(
            [0], dtype=np.int64
        )  # (int64) number of images seen during training
        # self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forward_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [0.83, 0.67]  # scales
        y = [
            self.forward_once(xi)[0]
            for xi in (
                x,
                torch_utils.scale_img(x.flip(3), s[0], same_shape=False),
                torch_utils.scale_img(x, s[1], same_shape=False),
            )
        ]
        y[1][..., :4] /= s[0]  # scale
        y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
        y[2][..., :4] /= s[1]  # scale

        # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
        #     area = yi[..., 2:4].prod(2)[:, :, None]
        #     if i == 1:
        #         yi *= (area < 96. ** 2).float()
        #     elif i == 2:
        #         yi *= (area > 32. ** 2).float()
        #     y[i] = yi

        y = torch.cat(y, 1)
        return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        if verbose:
            print("0", x.shape)
            str = ""

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat(
                (
                    x,
                    torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                    torch_utils.scale_img(x, s[1]),  # scale
                ),
                0,
            )

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in [
                "WeightedFeatureFusion",
                "FeatureConcat",
                "FeatureConcat2",
                "FeatureConcat3",
                "FeatureConcat_l",
            ]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [
                        list(out[i].shape) for i in module.layers
                    ]  # shapes
                    str = " >> " + " + ".join(["layer %g %s" % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print(
                    "%g/%g %s -" % (i, len(self.module_list), name), list(x.shape), str
                )
                str = ""

        if self.training:
            return yolo_out
        x, p = zip(*yolo_out)  # inference output, training output
        x = torch.cat(x, 1)  # cat yolo outputs
        if augment:  # de-augment results
            x = torch.split(x, nb, dim=0)
            x[1][..., :4] /= s[0]  # scale
            x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
            x[2][..., :4] /= s[1]  # scale
            x = torch.cat(x, 1)
        return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print("Fusing layers...")
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1 :])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        None if ONNX_EXPORT else self.info()  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == "darknet53.conv.74":
        cutoff = 75
    elif file == "yolov3-tiny.conv.15":
        cutoff = 15

    # Read weights file
    with open(weights, "rb") as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(
            f, dtype=np.int32, count=3
        )  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(
            f, dtype=np.int64, count=1
        )  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    # print(self.module_defs[:cutoff])
    # print(self.module_list[:cutoff])
    for mdef, module in zip(self.module_defs[:cutoff], self.module_list[:cutoff]):
        if mdef["type"] == "convolutional":
            conv = module[0]
            if mdef["batch_normalize"]:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.bias)
                )
                ptr += nb
                # Weight
                bn.weight.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.weight)
                )
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.running_mean)
                )
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.running_var)
                )
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr : ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
            ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(
                torch.from_numpy(weights[ptr : ptr + nw]).view_as(conv.weight)
            )
            ptr += nw


def parse_model_cfg(path):
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    if not path.endswith(".cfg"):  # add .cfg suffix if omitted
        path += ".cfg"
    if not os.path.exists(path) and os.path.exists(
        f"cfg{os.sep}{path}"
    ):  # add cfg/ prefix if omitted
        path = f"cfg{os.sep}{path}"

    with open(path, "r") as f:
        lines = f.read().split("\n")
    lines = [x for x in lines if x and not x.startswith("#")]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith("["):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]["type"] = line[1:-1].rstrip()
            if mdefs[-1]["type"] == "convolutional":
                mdefs[-1][
                    "batch_normalize"
                ] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == "anchors":  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape(
                    (-1, 2)
                )  # np anchors
            elif (key in ["from", "layers", "mask"]) or (
                key == "size" and "," in val
            ):  # return array
                mdefs[-1][key] = [int(x) for x in val.split(",")]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = (
                        int(val) if (int(val) - float(val)) == 0 else float(val)
                    )
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = [
        "type",
        "batch_normalize",
        "filters",
        "size",
        "stride",
        "pad",
        "activation",
        "layers",
        "groups",
        "from",
        "mask",
        "anchors",
        "classes",
        "num",
        "jitter",
        "ignore_thresh",
        "truth_thresh",
        "random",
        "stride_x",
        "stride_y",
        "weights_type",
        "weights_normalization",
        "scale_x_y",
        "beta_nms",
        "nms_kind",
        "iou_loss",
        "iou_normalizer",
        "cls_normalizer",
        "iou_thresh",
    ]

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(
        u
    ), f"Unsupported fields {u} in {path}. See https://github.com/ultralytics/yolov3/issues/631"

    return mdefs


def create_modules(module_defs, img_size, cfg):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = (
        [img_size] * 2 if isinstance(img_size, int) else img_size
    )  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        # print("i")
        # print(i)
        # print("mdef")
        # print(mdef)
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            stride = (
                mdef["stride"]
                if "stride" in mdef
                else (mdef["stride_y"], mdef["stride_x"])
            )
            if isinstance(k, int):  # single-size conv
                modules.add_module(
                    "Conv2d",
                    nn.Conv2d(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=k,
                        stride=stride,
                        padding=k // 2 if mdef["pad"] else 0,
                        groups=mdef["groups"] if "groups" in mdef else 1,
                        bias=not bn,
                    ),
                )
            else:  # multiple-size conv
                modules.add_module(
                    "MixConv2d",
                    MixConv2d(
                        in_ch=output_filters[-1],
                        out_ch=filters,
                        k=k,
                        stride=stride,
                        bias=not bn,
                    ),
                )

            if bn:
                modules.add_module(
                    "BatchNorm2d", nn.BatchNorm2d(filters, momentum=0.03, eps=1e-4)
                )
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if (
                mdef["activation"] == "leaky"
            ):  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            elif mdef["activation"] == "swish":
                modules.add_module("activation", Swish())
            elif mdef["activation"] == "mish":
                modules.add_module("activation", Mish())

        elif mdef["type"] == "maxpool":
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module("ZeroPad2d", nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module("MaxPool2d", maxpool)
            else:
                modules = maxpool

        elif mdef["type"] == "upsample":
            modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":  # nn.Sequential() placeholder for 'route' layer
            layers = mdef["layers"]
            filters = sum(output_filters[l + 1 if l > 0 else l] for l in layers)
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif (
            mdef["type"] == "shortcut"
        ):  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef["from"]
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(
                layers=layers, weight="weights_type" in mdef
            )

        elif mdef["type"] == "yolo":
            yolo_index += 1
            stride = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7 strides
            if any(
                x in cfg for x in ["yolov4-tiny", "fpn", "yolov3"]
            ):  # P5, P4, P3 strides
                stride = [32, 16, 8]
            layers = mdef["from"] if "from" in mdef else []
            modules = YOLOLayer(
                anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                nc=mdef["classes"],  # number of classes
                img_size=img_size,  # (416, 416)
                yolo_index=yolo_index,  # 0, 1, 2...
                layers=layers,  # output layers
                stride=stride[yolo_index],
            )

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if "from" in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[: modules.no * modules.na].view(
                    modules.na, -1
                )  # shape(3,85)
                # bias[:, 4] += -4.5  # obj
                bias[:, 4] += math.log(
                    8 / (640 / stride[yolo_index]) ** 2
                )  # obj (8 objects per 640 image)
                bias[:, 5:] += math.log(
                    0.6 / (modules.nc - 0.99)
                )  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(
                    bias_, requires_grad=bias_.requires_grad
                )
            except Exception:
                print("WARNING: smart bias initialization failure.")

        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class WeightedFeatureFusion(
    nn.Module
):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(
                torch.zeros(self.n), requires_grad=True
            )  # layer weights

    def forward(self, x, outputs):
        # print("WeightedFeatureFusion")
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = (
                outputs[self.layers[i]] * w[i + 1]
                if self.weight
                else outputs[self.layers[i]]
            )  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = (
                    x[:, :na] + a
                )  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class Swish(nn.Module):
    def forward(self, x):
        # print("Swish")
        return x * torch.sigmoid(x)


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * F.softplus(x).tanh()


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        # print("FeatureConcat")
        return (
            torch.cat([outputs[i] for i in self.layers], 1)
            if self.multiple
            else outputs[self.layers[0]]
        )


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (2)
        # print("nc")
        # print(nc)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        # if ONNX_EXPORT:
        #    self.training = False
        #    self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device=torch.device("cuda")):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        # if not self.training:
        yv, xv = torch.meshgrid(
            [torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)]
        )
        self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        # print("YOLOLayer")
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)
        # print("self.nx.ny")
        # print(self.nx)
        # print(self.ny)
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = (
            p.view(bs, self.na, self.no, self.ny, self.nx)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # prediction

        io = p.sigmoid()
        io[..., :2] = io[..., :2] * 2.0 - 0.5 + self.grid
        io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
        io[..., :4] *= self.stride
        # io = p.clone()  # inference output
        # io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
        # io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        # io[..., :4] *= self.stride
        # torch.sigmoid_(io[..., 4:])
        return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


def get_yolo_layers(model):
    return [
        i
        for i, m in enumerate(model.module_list)
        if m.__class__.__name__ == "YOLOLayer"
    ]  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == "darknet53.conv.74":
        cutoff = 75
    elif file == "yolov3-tiny.conv.15":
        cutoff = 15

    # Read weights file
    with open(weights, "rb") as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(
            f, dtype=np.int32, count=3
        )  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(
            f, dtype=np.int64, count=1
        )  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for mdef, module in zip(self.module_defs[:cutoff], self.module_list[:cutoff]):
        if mdef["type"] == "convolutional":
            conv = module[0]
            if mdef["batch_normalize"]:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.bias)
                )
                ptr += nb
                # Weight
                bn.weight.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.weight)
                )
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.running_mean)
                )
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.running_var)
                )
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr : ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
            ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(
                torch.from_numpy(weights[ptr : ptr + nw]).view_as(conv.weight)
            )
            ptr += nw


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
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
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


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [
        unique_classes.shape[0],
        tp.shape[1],
    ]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_gt + 1e-16)  # recall curve
        r[ci] = np.interp(
            -pr_score, -conf[i], recall[:, 0]
        )  # r at pr_score, negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def output_to_target(output, width, height):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    # if isinstance(output, torch.Tensor):
    #    output = output.cpu().numpy()
    # output = output.cpu().numpy()
    # print(output)

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = ((box[2] - box[0]) / width).cpu().numpy()
                h = ((box[3] - box[1]) / height).cpu().numpy()
                x = (box[0] / width + w / 2).cpu().numpy()
                y = (box[1] / height + h / 2).cpu().numpy()
                conf = pred[4].cpu().numpy()
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    eps = 1e-9
    EIoU = False
    ECIoU = False

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
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if not GIoU and not DIoU and not CIoU and not EIoU and not ECIoU:
        return iou  # IoU
    cw = torch.max(b1_x2, b2_x2) - torch.min(
        b1_x1, b2_x1
    )  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    if (
        CIoU or DIoU or EIoU or ECIoU
    ):  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
        c2 = cw**2 + ch**2 + eps  # convex diagonal squared
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
        ) / 4  # center distance squared
        if DIoU:
            return iou - rho2 / c2  # DIoU
        elif (
            CIoU
        ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
            v = (4 / math.pi**2) * torch.pow(
                torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
            )
            with torch.no_grad():
                alpha = v / ((1 + eps) - iou + v)
            return iou - (rho2 / c2 + v * alpha)  # CIoU
        elif EIoU:  # Efficient IoU https://arxiv.org/abs/2101.08158
            rho3 = (w1 - w2) ** 2
            c3 = cw**2 + eps
            rho4 = (h1 - h2) ** 2
            c4 = ch**2 + eps
            return iou - rho2 / c2 - rho3 / c3 - rho4 / c4  # EIoU
        else:
            v = (4 / math.pi**2) * torch.pow(
                torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
            )
            with torch.no_grad():
                alpha = v / ((1 + eps) - iou + v)
            rho3 = (w1 - w2) ** 2
            c3 = cw**2 + eps
            rho4 = (h1 - h2) ** 2
            c4 = ch**2 + eps
            return iou - v * alpha - rho2 / c2 - rho3 / c3 - rho4 / c4  # ECIoU
    else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [min(recall[-1] + 1e-3, 1.0)]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        return np.trapz(np.interp(x, mrec, mpre), x)
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def non_max_suppression(
    prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False
):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            try:
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except Exception:
                print(x, i, x.shape, i.shape)
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def save_summary(opt, run, best_train_acc, best_val_acc):
    # Generate summary
    summary = {
        "Number of epoch": [],
        "Best train accuarcy": [],
        "Best validation accuarcy": [],
    }
    if opt.mode == "train":
        summary["Number of epoch"] = run
        summary["Best train accuarcy"] = round(best_train_acc, 4)
        summary["Best validation accuarcy"] = round(best_val_acc, 4)
    if opt.mode == "val":
        summary["Best validation accuarcy"] = round(best_val_acc, 4)

    if not os.path.exists(os.path.dirname(opt.out_dir)):
        os.makedirs(os.path.dirname(opt.out_dir))

    # summary_file = opt.out_dir + '/result/' + opt.mode + '_summary.yaml'
    summary_file = os.path.join(opt.out_dir, "result", f"{opt.mode}_summary.yaml")

    if os.path.exists(summary_file):
        os.remove(summary_file)

    with open(summary_file, "w") as outfile:
        yaml.dump(summary, outfile)


def save_config_yaml(opt):
    output = {
        "framework_type": ["Pytorch"],
        "input_type": ["Single Image"],
        "model_type": ["YOLO"],
        "img_size": opt.img_size,
        "transforms": {},
        "output_type": ["Detection"],
        "category_list": [opt.categpry_dic],
    }
    output_config = os.path.join(opt.model_dir, f"{opt.name}config.yaml")
    if not os.path.exists(os.path.dirname(output_config)):
        os.makedirs(os.path.dirname(output_config))
    with open(output_config, "w") as outfile:
        yaml.dump(output, outfile)


def test(
    data,
    weights=None,
    batch_size=16,
    imgsz=640,
    conf_thres=0.05,
    # conf_thres=0.0001,
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
    opt=None,
):
    # Initialize/load model and set device
    training = model is not None
    # initial val_out.yaml

    device = select_device(opt.device, batch_size=batch_size)
    # device = torch.device("cuda")
    # merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
    save_txt = False  # use Merge NMS, save *.txt labels

    model = torch.load(opt.weights, map_location=device)

    model.eval()

    nc = class_num
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    img = torch.zeros((1, 3, imgsz[0], imgsz[1]), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    _ = model(img)  # run once
    dataloader = create_dataloader(
        data,
        imgsz[0],
        batch_size,
        32,
        opt,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.5,
        rect=False,
    )[0]
    # test use rec
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
    p, r, f1, mp, mr, map50, map, t0, t1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for img, targets, paths, shapes in tqdm(dataloader, desc=s):
        # print(paths)

        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            # t = time_synchronized()
            print("img shape before model python")
            print(img.shape)
            inf_out, train_out = model(img)  # inference and training outputs
            # t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][
                    :3
                ]  # GIoU, obj, cls

            # Run NMS
            # t = time_synchronized()
            output = non_max_suppression(
                inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=False
            )
            # t1 += time_synchronized() - t

        sample_data = output_to_target(output, width, height)
        # print("after out put to target")
        # print(sample_data)
        # sample_data_1 = [item for item in sample_data if item[6] > conf_thres]

        for si, pred in enumerate(output):
            # print("output")
            # print(output)
            labels = targets[targets[:, 0] == si, 1:]
            # print("labels")
            # print(labels)
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

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="train", help="train or val or test"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output/model/",
        help="path to store model file",
    )
    parser.add_argument(
        "--name", default="", help="renames results.txt to results_name.txt if supplied"
    )
    # training parameters
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="../data/val_AT.lmdb",
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--category_file",
        type=str,
        default="../data/category.yaml",
        help="the absolute path of the category file",
    )
    # parser.add_argument('--weights', type=str, default="symbols/yolov4-csp.weights", help='initial pretrain weights(use default) path or load weights during testing, leave to None if train from scratch')
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="initial pretrain weights(use default) path or load weights during testing, leave to None if train from scratch",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=[320, 320], help="train,test sizes"
    )
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=[0], help="GPUs to be used"
    )
    # testing parameters
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.05,
        help="object confidence threshold during test",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.65,
        help="IOU threshold for NMS during test",
    )

    opt = parser.parse_args()
    opt.device = str(opt.gpu_ids[0])

    with open(opt.category_file, "r") as stream_cat:
        doc_cat = yaml.safe_load(stream_cat)

    # agreed from AT read categories files, ignore first head and sore in nature order
    opt.categpry_dic = []
    opt.categpry_dic.extend(str(item["id"]) for item in doc_cat["categories"][1:])
    save_config_yaml(opt)

    # print("caty ids")
    # print(opt.categpry_dic)

    opt.class_number = len(opt.categpry_dic)

    opt.single_cls = opt.class_number == 1  # DDP parameter, do not modify

    # yolov4-csp_class80_swish.cfg
    # yolov4-csp.cfg

    results_val, _ = test(
        opt.val_dataset,
        opt.weights,
        opt.batch_size,
        opt.img_size,
        opt.conf_thres,
        opt.iou_thres,
        False,
        opt.single_cls,
        False,
        False,
        class_num=opt.class_number,
        opt=opt,
    )

    print("val mAP")
    print(results_val[2])
    # save_summary(opt, 0, 0, float(results_val[2]))


# TODO:
