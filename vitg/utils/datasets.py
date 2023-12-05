import glob
import math
import os
import pickle
import random
import shutil
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from PIL import ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from vitg.network.backbone.vitgyolov7.utils.general import xywhn2xyxy
from vitg.utils.general import torch_distributed_zero_first, xyxy2xywh

help_url = ""
img_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
vid_formats = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:
            s = (s[1], s[0])
    except Exception:
        pass

    return s


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
    with torch_distributed_zero_first(local_rank):
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
        drop_last=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    return dataloader, dataset


def create_dataloader_squaretest(
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
    with torch_distributed_zero_first(local_rank):
        dataset = LoadImagesAndLabels_squaretest(
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


def create_dataloader_fortest2(
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
    # print("stride value in loader")
    # print(stride)
    # print("padding value in loader")
    # print(pad)
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(local_rank):
        dataset = LoadImagesAndLabels_test(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            single_cls=opt.single_cls,
            stride=int(stride),
            pad=pad,
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
        # collate_fn=None
        collate_fn=LoadImagesAndLabels_test.collate_fn,
    )
    return dataloader, dataset


def create_dataloader_noletterbox(
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
    with torch_distributed_zero_first(local_rank):
        dataset = LoadImagesAndLabels_noletterbox(
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
        drop_last=True,
        collate_fn=LoadImagesAndLabels_noletterbox.collate_fn,
    )
    return dataloader, dataset




def unpack_img2(buf, iscolor=1):
    img = np.frombuffer(buf, dtype=np.uint8)
    assert cv2 is not None
    img = cv2.imdecode(img, iscolor)
    return img


def unpack_img_lmdb(buf, iscolor=1):
    img = np.frombuffer(buf, dtype=np.uint8)
    assert cv2 is not None
    img = cv2.imdecode(img, iscolor)
    return img


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
        stride=64,
        pad=0.0,
        opt=None,
    ):
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
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride

        shapes_off = []
        for index in range(self.length):
            with self.env.begin(write=False) as txn:
                byteflow_shape = txn.get(self.keys[index])
            unpacked_shape = pickle.loads(byteflow_shape)
            shapes_off.append([unpacked_shape[3], unpacked_shape[4]])
        self.shapes = np.array(shapes_off)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  #
            # print(s)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
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
        # return self.length
        return 30

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
                img2, labels2 = load_mosaic(self, random.randint(0, self.length - 1))
                # img2, labels2 = load_mosaic(self, random.randint(0, len(self.samples) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            # Letterbox
            shape = (
                self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            )  # final letterboxed shape

            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            unpacked_labels_nomosaic = [
                [
                    item["category_id"][0],
                    item["bbox"][0] + item["bbox"][2] / 2,
                    item["bbox"][1] + item["bbox"][3] / 2,
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

        # print("return img shape")
        # print(torch.from_numpy(img).shape)
        # print("lables out shape")
        # print(labels_out.shape)
        # return torch.from_numpy(img), labels_out, self.img_files[index], shapes
        return torch.from_numpy(img), labels_out, self.unpacked[5], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


class LoadImagesAndLabels_noletterbox(Dataset):  # for training/testing
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
        stride=64,
        pad=0.0,
        opt=None,
    ):
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

        shapes_off = []
        for index in range(self.length):
            with self.env.begin(write=False) as txn:
                byteflow_shape = txn.get(self.keys[index])
            unpacked_shape = pickle.loads(byteflow_shape)
            shapes_off.append([unpacked_shape[3], unpacked_shape[4]])
        self.shapes = np.array(shapes_off)

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
        # return self.length
        return 30

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
        # print("mosaic check")
        # print(self.mosaic)
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp["mixup"]:
                img2, labels2 = load_mosaic(self, random.randint(0, self.length - 1))
                # img2, labels2 = load_mosaic(self, random.randint(0, len(self.samples) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image_sqr(self, index)

            # Letterbox
            shape = self.img_size  # final letterboxed shape

            img, ratio, pad = letterbox_disable(
                img, shape, auto=False, scaleup=self.augment
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            unpacked_labels_nomosaic = [
                [
                    item["category_id"][0],
                    item["bbox"][0] + item["bbox"][2] / 2,
                    item["bbox"][1] + item["bbox"][3] / 2,
                    item["bbox"][2],
                    item["bbox"][3],
                ]
                for item in self.unpacked[1]
            ]
            # unpacked_labels_nomosaic=[[self.categpry_dic.index(int(item["category_id"][0])),item["bbox"][0],item["bbox"][1],item["bbox"][2],item["bbox"][3]] for item in self.unpacked[1]]
            x = np.array(unpacked_labels_nomosaic, dtype=np.float)

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

        return torch.from_numpy(img), labels_out, self.unpacked[5], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes




class LoadImagesAndLabels_test(Dataset):  # for training/testing
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
        stride=1,
        pad=0.0,
        opt=None,
    ):
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
        # print("self batch")
        # print(self.batch)
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
        self.pad = pad

    def __len__(self):
        # return self.length
        return 30

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
                img2, labels2 = load_mosaic(self, random.randint(0, self.length - 1))
                # img2, labels2 = load_mosaic(self, random.randint(0, len(self.samples) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            # print("load img without mosaic")
            # print(index)
            img, (h0, w0), (h, w) = load_image(self, index)
            # print("size after load and img sum1")
            # print(img.shape)
            # print((h0, w0))
            # print((h, w))
            # print(img.sum())

            shapes = [[h0 / w0, 1]] if w0 >= h0 else [[1, w0 / h0]]
            # print("shape roti")
            # print(shapes)
            stride2 = self.stride
            pad2 = self.pad
            # fix stride and pad value for proper box location
            shape0 = (
                np.ceil(np.array(shapes) * self.img_size / stride2 + pad2).astype(
                    np.int
                )
                * stride2
            )
            shape = shape0[0] if self.rect else self.img_size

            # Letterbox
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            unpacked_labels_nomosaic = [
                [
                    item["category_id"][0],
                    item["bbox"][0] + item["bbox"][2] / 2,
                    item["bbox"][1] + item["bbox"][3] / 2,
                    item["bbox"][2],
                    item["bbox"][3],
                ]
                for item in self.unpacked[1]
            ]
            # unpacked_labels_nomosaic=[[self.categpry_dic.index(int(item["category_id"][0])),item["bbox"][0],item["bbox"][1],item["bbox"][2],item["bbox"][3]] for item in self.unpacked[1]]
            x = np.array(unpacked_labels_nomosaic, dtype=np.float)

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


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image_sqr(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    # img = self.imgs[index]

    img = unpack_img_lmdb(self.unpacked[0])

    # img = self.unpack_img(self.samples[index][0])

    h0, w0 = img.shape[:2]  # orig hw
    # print("inter load img imgsize")
    # print(self.img_size)
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        # img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=interp)
    return (
        img,
        (self.img_size, self.img_size),
        img.shape[:2],
    )  # img, hw_original, hw_resized


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image_index_mosic_andlabel(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    # img = self.imgs[index]

    with self.env.begin(write=False) as txn:
        byteflow_mosic_local = txn.get(self.keys[index])
    unpacked_mosic_local = pickle.loads(byteflow_mosic_local)

    img = unpack_img_lmdb(unpacked_mosic_local[0])

    unpacked_labels_mosic_local = [
        [
            item["category_id"][0],
            item["bbox"][0] + item["bbox"][2] / 2,
            item["bbox"][1] + item["bbox"][3] / 2,
            item["bbox"][2],
            item["bbox"][3],
        ]
        for item in unpacked_mosic_local[1]
    ]

    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return (
        img,
        (h0, w0),
        img.shape[:2],
        unpacked_labels_mosic_local,
    )  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    # yolov4csp mosaic
    yc, xc = s, s  # mosaic center x, y
    # yolor mosaic
    # yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [
        random.randint(0, self.length - 1) for _ in range(3)
    ]  # 3 additional image indices

    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w), label_mosic_localx = load_image_index_mosic_andlabel(
            self, index
        )
        # place img in img4
        if i == 0:  # top left
            img4 = np.full(
                (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
            )  # base image with 4 tiles
            x1a, y1a, x2a, y2a = (
                max(xc - w, 0),
                max(yc - h, 0),
                xc,
                yc,
            )  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                h - (y2a - y1a),
                w,
                h,
            )  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # yolov4csp mosaic
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            # yolor mosaic
            # x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = np.array(label_mosic_localx, dtype=np.float)

        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    img4, labels4 = random_perspective(
        img4,
        labels4,
        degrees=self.hyp["degrees"],
        translate=self.hyp["translate"],
        scale=self.hyp["scale"],
        shear=self.hyp["shear"],
        perspective=self.hyp["perspective"],
        border=self.mosaic_border,
    )  # border to remove

    return img4, labels4


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(
            random.uniform(0, w - bw)
        )  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
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
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
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


def letterbox_disable(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
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
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
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


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def box_candidates(
    box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2
):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
            np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
        ).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = (
        [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
    )  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(
    path="path/images", img_size=1024
):  # from utvitg.utilsils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = f"{path}_reduced"
    create_folder(path_new)
    for f in tqdm(glob.glob(f"{path}/*.*")):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(
                    img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA
                )  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except Exception:
            print(f"WARNING: image failure {f}")


def recursive_dataset2bmp(
    dataset="path/dataset_bmp",
):  # from vitg.utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = f"{a}/{file}"
            s = Path(file).suffix
            if s == ".txt":  # replace text
                with open(p, "r") as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, ".bmp")
                with open(p, "w") as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, ".bmp"), cv2.imread(p))
                if s != ".bmp":
                    os.system(f"rm '{p}'")


def imagelist2folder(
    path="path/images.txt",
):  # from vitg.utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, "r") as f:
        for line in f.read().splitlines():
            os.system(f'cp "{line}" {path[:-4]}')
            print(line)


def create_folder(path="./new"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
