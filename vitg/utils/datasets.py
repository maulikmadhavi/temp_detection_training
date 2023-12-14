import os
import pickle
import random

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from vitg.loader.transforms_lib.transform_utils import (
    augment_hsv,
    letterbox,
    letterbox_disable,
    load_image,
    load_image_sqr,
    load_mosaic,
    random_perspective,
)


from vitg.utils.general import torch_distributed_zero_first, xyxy2xywh


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
        drop_last=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    return dataloader, dataset


def create_test_dataloader(
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
        dataset = _LoadImagesAndLabelsTest(
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
        collate_fn=_LoadImagesAndLabelsTest.collate_fn,
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
        dataset = _LoadImagesAndLabelsNoletterbox(
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
        drop_last=True,
        collate_fn=_LoadImagesAndLabelsNoletterbox.collate_fn,
    )
    return dataloader, dataset


# ============ Load Images and Labels ============
class LoaderBaseClass(Dataset):
    @staticmethod
    def __init__(
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        cache_images=False,
        single_cls=False,
        stride=64,
        pad=0.0,
    ):
        pass

    @staticmethod
    def __len__():
        return 30

    @staticmethod
    def collate_fn(batch):
        (img, label, path, shapes) = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return (torch.stack(img, 0), torch.cat(label, 0), path, shapes)

    def normxywh_to_pixelxyxy(self, h, w, ratio, pad, x):
        labels = x.copy()
        labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
        labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
        labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
        labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        return labels

    def get_env_len(self, path):
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


class LoadImagesAndLabels(LoaderBaseClass):  # for training/testing
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
        self.get_env_len(path)
        self.batch = np.floor(np.arange(self.length) / batch_size).astype(np.int16)
        nb = self.batch[-1] + 1

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

        # Get the shapes of all images
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
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[self.batch == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int16)
                * stride
            )

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
            x = np.array(unpacked_labels_nomosaic, dtype=np.float16)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = self.normxywh_to_pixelxyxy(h, w, ratio, pad, x)
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
        n_l = len(labels)  # number of labels
        if n_l:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1
        if self.augment:
            # flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if n_l:
                    labels[:, 2] = 1 - labels[:, 2]
            # flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if n_l:
                    labels[:, 1] = 1 - labels[:, 1]
        labels_out = torch.zeros((n_l, 6))
        if n_l:
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.unpacked[5], shapes


class _LoadImagesAndLabelsNoletterbox(LoaderBaseClass):
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
        self.get_env_len(path)

        self.unpacked = []
        self.batch = np.floor(np.arange(self.length) / batch_size).astype(np.int16)
        nb = self.batch[-1] + 1
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and (not self.rect)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        shapes_off = []
        for index in range(self.length):
            with self.env.begin(write=False) as txn:
                byteflow_shape = txn.get(self.keys[index])
            unpacked_shape = pickle.loads(byteflow_shape)
            shapes_off.append([unpacked_shape[3], unpacked_shape[4]])
        self.shapes = np.array(shapes_off)
        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.shapes = s[irect]
            ar = ar[irect]
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[self.batch == i]
                (mini, maxi) = (ari.min(), ari.max())
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int16)
                * stride
            )

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        self.unpacked = pickle.loads(byteflow)
        if self.image_weights:
            index = self.indices[index]
        hyp = self.hyp
        if self.mosaic:
            (img, labels) = load_mosaic(self, index)
            shapes = None
            if random.random() < hyp["mixup"]:
                (img2, labels2) = load_mosaic(self, random.randint(0, self.length - 1))
                r = np.random.beta(8.0, 8.0)
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            (img, (h0, w0), (h, w)) = load_image_sqr(self, index)
            shape = self.img_size
            (img, ratio, pad) = letterbox_disable(
                img, shape, auto=False, scaleup=self.augment
            )
            shapes = ((h0, w0), ((h / h0, w / w0), pad))
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
            x = np.array(unpacked_labels_nomosaic, dtype=np.float16)
            if x.size > 0:
                labels = self.normxywh_to_pixelxyxy(h, w, ratio, pad, x)

        if self.augment:
            if not self.mosaic:
                (img, labels) = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])
        n_l = len(labels)
        if n_l:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]
        if self.augment:
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if n_l:
                    labels[:, 2] = 1 - labels[:, 2]
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if n_l:
                    labels[:, 1] = 1 - labels[:, 1]
        labels_out = torch.zeros((n_l, 6))
        if n_l:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return (torch.from_numpy(img), labels_out, self.unpacked[5], shapes)


class _LoadImagesAndLabelsTest(LoaderBaseClass):
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
        self.get_env_len(path)
        self.unpacked = []

        self.batch = np.floor(np.arange(self.length) / batch_size).astype(np.int16)
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and (not self.rect)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.pad = pad

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        self.unpacked = pickle.loads(byteflow)
        if self.image_weights:
            index = self.indices[index]
        hyp = self.hyp
        if self.mosaic:
            (img, labels) = load_mosaic(self, index)
            shapes = None
            if random.random() < hyp["mixup"]:
                (img2, labels2) = load_mosaic(self, random.randint(0, self.length - 1))
                r = np.random.beta(8.0, 8.0)
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            (img, (h0, w0), (h, w)) = load_image(self, index)
            shapes = [[h0 / w0, 1]] if w0 >= h0 else [[1, w0 / h0]]
            stride2 = self.stride
            pad2 = self.pad
            shape0 = (
                np.ceil(np.array(shapes) * self.img_size / stride2 + pad2).astype(
                    np.int16
                )
                * stride2
            )
            shape = shape0[0] if self.rect else self.img_size
            (img, ratio, pad) = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = ((h0, w0), ((h / h0, w / w0), pad))
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
            x = np.array(unpacked_labels_nomosaic, dtype=np.float16)
            if x.size > 0:
                labels = self.normxywh_to_pixelxyxy(h, w, ratio, pad, x)
        if self.augment:
            if not self.mosaic:
                (img, labels) = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

        n_l = len(labels)
        if n_l:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]
        if self.augment:
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if n_l:
                    labels[:, 2] = 1 - labels[:, 2]
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if n_l:
                    labels[:, 1] = 1 - labels[:, 1]
        labels_out = torch.zeros((n_l, 6))
        if n_l:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return (torch.from_numpy(img), labels_out, self.unpacked[5], shapes)
