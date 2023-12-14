import math
import os
import torch
import torch.distributed as dist
from vitg.utils.general import check_img_size
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml


def get_hyp(config):
    with open(config.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    hyp_keys = ["fl_gamma", "translate", "flipud", "mixup", "degrees", "scale"]
    hyp.update({k: getattr(config, k) for k in hyp_keys})
    hyp["lr0"] = config.lr
    hyp["mosaic"] = "mosaic" in config.transforms_train

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
    config.device = (
        ",".join(str(i) for i in range(config.world_size))
        if config.use_ddp
        else str(config.gpu_ids[0])
    )
    if config.use_ddp:
        print("all device use")
        print(config.device)

    config.weights = config.pretrained_model
    get_arch_specific_hyperparams(config)

    # test free image size
    imgsz, imgsz_test = [
        check_img_size(x, config.gs) for x in config.img_size
    ]  # verify imgsz are gs-multiples

    config.img_size = [imgsz, imgsz]
    config.total_batch_size = config.batch_size

    if config.mode == "train":
        config.global_rank = int(os.getenv("RANK", -1))
        if config.local_rank != -1:
            run_once_for_ddp(config)
    config.single_cls = config.class_number == 1


# TODO Rename this here and in `modified_parameters`
def run_once_for_ddp(config):
    assert torch.cuda.device_count() > config.local_rank
    torch.cuda.set_device(config.local_rank)
    device = torch.device("cuda", config.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    config.global_rank = dist.get_rank()
    assert (
        config.batch_size % config.world_size == 0
    ), "--batch-size must be multiple of CUDA device count"
    config.batch_size = config.total_batch_size // config.world_size


def get_arch_specific_hyperparams(config):
    arch_params = {
        "yolov7": (
            64,
            "vitg/symbols/yolov7-e6e.yaml",
            "vitg/data/hyp.yolov7.scratch.custom_nolr.yaml",
        ),
        "yolov4csp": (
            32,
            "vitg/symbols/yolov4-csp_class80_swish.cfg",
            "vitg/data/hyp.scratch_nolr_v2.yaml",
        ),
        "yolor": (
            64,
            "vitg/symbols/yolor-d6.yaml",
            "vitg/data/hyp.yolor.finetune.1280_nolr.yaml",
        ),
        "mobilenetssd": (1, "", "vitg/data/hyp.yolor.finetune.1280_nolr.yaml"),
        "yolov8": (
            1,
            "vitg/symbols/yolov8x.yaml",
            "vitg/data/hyp.yolor.finetune.1280_nolr.yaml",
        ),
    }
    if config.arch not in arch_params:
        raise ValueError(f"Architecture '{config.arch}' not supported")
    config.gs, config.cfg, config.hyp = arch_params[config.arch]
    if config.arch == "mobilenetssd":
        config.img_size = [300, 300]
        config.imgsz = config.imgsz_test = 300


def get_cateDict_classNum(config):
    with open(config.category_file, "r") as stream_cat:
        doc_cat = yaml.load(stream_cat, Loader=yaml.FullLoader)

    config.category_dic = [str(item["id"]) for item in doc_cat["categories"][1:]]
    config.class_number = len(config.category_dic)


def get_optimizer_scheduler(hyp, epochs, net):
    optimizer = optim.SGD(
        net.pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
    )
    optimizer.add_param_group({"params": net.pg1, "weight_decay": hyp["weight_decay"]})
    optimizer.add_param_group({"params": net.pg2})

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return optimizer, lf, scheduler
