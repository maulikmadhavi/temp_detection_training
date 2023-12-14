import math
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml

from vitg.utils.general import check_img_size


def get_hyp(config):
    """Extract hyperparameters from the config

    Args:
        config (_type_): Namespace

    Returns:

    """
    with open(config.hyp) as f:
        # hyp = yaml.safe_load(f)  # load hyps
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        # update learning rate

    hyp.update(lr0=config.lr)
    hyp.update(fl_gamma=config.fl_gamma)
    hyp.update(translate=config.translate)
    hyp.update(flipud=config.flipud)
    hyp.update(mixup=config.mixup)
    hyp.update(degrees=config.degrees)
    hyp.update(scale=config.scale)
    hyp.update(mosaic=True)

    all_transofmritems_num = ["translate", "flipud", "mixup", "degrees", "scale"]
    for transformitem in all_transofmritems_num:
        if transformitem not in config.transforms_train:
            hyp[transformitem] = 0
    all_transofmritems_binery = ["mosaic"]
    for transformitem in all_transofmritems_binery:
        if transformitem not in config.transforms_train:
            hyp.update(mosaic=False)

    return hyp


def modified_parameters(config):
    if not config.use_ddp:
        config.device = str(config.gpu_ids[0])
    else:
        config.device = "0"
        for index in range(config.world_size - 1):
            config.device += f",{str(index + 1)}"
        print("all device use")
        print(config.device)
    # config.img-size=config.img_size
    # config.batch-size=config.batch_size
    config.weights = config.pretrained_model

    get_arch_specific_hyperparams(config)

    # test free image size
    imgsz, imgsz_test = [
        check_img_size(x, config.gs) for x in config.img_size
    ]  # verify imgsz are gs-multiples

    config.img_size = [imgsz, imgsz]
    config.total_batch_size = config.batch_size

    if config.mode == "train":
        config.img_size.extend(
            [config.img_size[-1]] * (2 - len(config.img_size))
        )  # extend to 2 sizes (train, test)

        config.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
        # DDP mode
        if config.local_rank != -1:
            assert torch.cuda.device_count() > config.local_rank
            torch.cuda.set_device(config.local_rank)
            device = torch.device("cuda", config.local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )  # distributed backend

            config.global_rank = dist.get_rank()
            assert (
                config.batch_size % config.world_size == 0
            ), "--batch-size must be multiple of CUDA device count"
            config.batch_size = config.total_batch_size // config.world_size

    config.single_cls = config.class_number == 1  # DDP parameter, do not modify


def get_arch_specific_hyperparams(config):
    """This function will extract the architecture specific hyper parameters"""
    if config.arch == "yolov7":
        config.gs = 64
        config.cfg = (
            "vitg/symbols/yolov7-e6e.yaml"  # store yolo model structure(use default)
        )
        config.hyp = "vitg/data/hyp.yolov7.scratch.custom_nolr.yaml"  # store hyper parameters like GIoU loss gain, cls loss gain ， obj loss gain(use default)
    elif config.arch == "yolov4csp":
        config.gs = 32  # grid size (max stride)
        config.cfg = "vitg/symbols/yolov4-csp_class80_swish.cfg"  # store yolo model structure(use default)
        config.hyp = "vitg/data/hyp.scratch_nolr_v2.yaml"  # store hyper parameters like GIoU loss gain, cls loss gain ， obj loss gain(use default)
    elif config.arch == "yolor":
        config.gs = 64  # grid size (max stride)
        config.cfg = (
            "vitg/symbols/yolor-d6.yaml"  # store yolo model structure(use default)
        )
        config.hyp = "vitg/data/hyp.yolor.finetune.1280_nolr.yaml"  # store hyper parameters like GIoU loss gain, cls loss gain ， obj loss gain(use default)
    elif config.arch == "mobilenetssd":
        config.img_size = [300, 300]
        config.gs = 1  # grid size (max stride)
        config.cfg = ""  # store yolo model structure(use default)
        config.hyp = "vitg/data/hyp.yolor.finetune.1280_nolr.yaml"  # store hyper parameters like GIoU loss gain, cls loss gain ， obj loss gain(use default)
    elif config.arch == "yolov8":
        config.gs = 1  # grid size (max stride)
        config.cfg = (
            "vitg/symbols/yolov8x.yaml"  # store yolo model structure(use default)
        )
        config.hyp = "vitg/data/hyp.yolor.finetune.1280_nolr.yaml"  # store hyper parameters like GIoU loss gain, cls loss gain ， obj loss gain(use default)
    else:
        print("arch not supported")
        # print(type(config.arch))
        print(config.arch)
        exit()


def get_cateDict_classNum(config):
    """This function will prepare the category dict and get the class number"""
    with open(config.category_file, "r") as stream_cat:
        doc_cat = yaml.load(stream_cat, Loader=yaml.FullLoader)
        # doc_cat = yaml.safe_load(stream_cat)

    # agreed from AT read categories files, ignore first head and sore in nature order
    config.category_dic = []
    config.category_dic.extend(str(item["id"]) for item in doc_cat["categories"][1:])
    config.class_number = len(config.category_dic)


def get_optimizer_scheduler(hyp, epochs, net):
    optimizer = optim.SGD(
        net.pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
    )

    optimizer.add_param_group(
        {"params": net.pg1, "weight_decay": hyp["weight_decay"]}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": net.pg2})  # add pg2 (biases)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = (
        lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return optimizer, lf, scheduler
