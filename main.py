import argparse

# from solver_metic import Solver
import os
from distutils.util import strtobool

# from vitg.utils.general import increment_dir
from solver import Solver


def parse() -> argparse.Namespace:
    """Parse the input arguments"""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or val or test",
        choices=["train", "test", "val"],
    )

    # Dataset parameters
    group = parser.add_argument_group("Dataset parameters")
    group.add_argument(
        "--train_dataset",
        type=str,
        default="../data/train.lmdb",
        help="Path to training dataset",
    )
    group.add_argument(
        "--train_dataset_name", type=str, default="", help="Name of training dataset"
    )
    group.add_argument(
        "--val_dataset",
        type=str,
        default="../data/val.lmdb",
        help="Path to validation dataset",
    )
    group.add_argument(
        "--val_dataset_name", type=str, default="", help="Name of validation dataset"
    )
    group.add_argument(
        "--test_dataset",
        type=str,
        default="../data/val.lmdb",
        help="Path to test dataset",
    )
    group.add_argument(
        "--test_dataset_name", type=str, default="", help="Name of the dataset"
    )
    group.add_argument(
        "--category_file",
        type=str,
        default="../data/category.yaml",
        help="the absolute path of the category file",
    )

    # Model parameters
    group = parser.add_argument_group("Model parameters")
    # parser.add_argument('--weights', type=str, default="symbols/yolov4-csp.weights", help='initial pretrain weights(use default) path or load weights during testing, leave to None if train from scratch')
    group.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        help="initial pretrain weights(use default) path or load weights during testing, leave to None if train from scratch",
    )
    group.add_argument(
        "--arch",
        type=str,
        default="yolov4csp",
        choices=["yolov4csp", "yolov7", "yolor", "mobilenetssd", "yolov8"],
        help="architecture use in training",
    )
    group.add_argument(
        "--backbone",
        type=str,
        default="darknet",
        choices=["darknet"],
        help="backbone use in training",
    )
    group.add_argument(
        "--weight_initialization",
        type=str,
        default="normal",
        choices=["kaiming", "xavier", "normal"],
        help="Type of Weight Initialization",
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="./output/log/log.txt",
        help="path of the log file",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./output/", help="path of the log file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output/model/",
        help="path to store model file",
    )
    # parser.add_argument('--class-number', type=int, default=80, help='number of classes')
    parser.add_argument("--epochs", type=int, default=50, help="max epoch to train")
    parser.add_argument(
        "--early_stop", type=int, default=50, help="epochs to early stopping"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--yaml_batch_size",
        type=int,
        default=4,
        help="batch size to dump into output yaml",
    )
    parser.add_argument(
        "--img_size", nargs="+", type=int, default=[320, 320], help="train,test sizes"
    )
    # parser.add_argument('--gpu_ids', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=[0], help="GPUs to be used"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs/",
        help="logging directory and model save directory",
    )
    parser.add_argument(
        "--name", default="", help="renames results.txt to results_name.txt if supplied"
    )
    # parser.add_argument('--load_checkpoint',type=bool,default=False,help='turn on/off loading checkpoint')

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./output/checkpoint/",
        help="path of the checkpoints",
    )
    # parser.add_argument('--load_checkpoint', type=bool, default=False, help='turn on/off loading checkpoint')
    parser.add_argument(
        "--checkpoint_step", type=int, default=2, help="checkpoint step"
    )
    parser.add_argument(
        "--checkpoint_file", type=str, default="", help="path to checkpoint file"
    )
    parser.add_argument(
        "--load_checkpoint",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="whether to load checkpoint, temp method",
    )

    parser.add_argument(
        "--save_c_model",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Turn on/off saving C model",
    )
    parser.add_argument("--debug", action="store_true", help="use debug mode")
    parser.add_argument(
        "--use_tensorboard",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Turn on/off tensorboard",
    )

    parser.add_argument(
        "--fl_gamma",
        type=float,
        default=0.0,
        help="focal loss gamma, efficientDet default gamma=1.5",
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=0.0,
        help="image agument image translate, default is 0 which is disabled, range 0~1",
    )
    parser.add_argument(
        "--flipud",
        type=float,
        default=0.0,
        help="percentage of random flip up and down, range 0~1",
    )
    parser.add_argument(
        "--mixup", type=float, default=0.0, help="percentage of mixup, range 0~1"
    )
    parser.add_argument(
        "--degrees", type=float, default=0.0, help="# image rotation (+/- deg)"
    )
    parser.add_argument(
        "--scale", type=float, default=0.5, help="# image scale (+/- gain)"
    )
    # parser.add_argument('--mosaic',type=lambda x: bool(strtobool(x)),default=False,help='image mosaic for training')

    # testing parameters
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.05,
        help="object confidence threshold during test",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.65,
        help="IOU threshold for NMS during test",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="report mAP by class during test"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="save test results to *.txt during test"
    )
    parser.add_argument(
        "--use_amp",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Turn on/off Mixed precision training",
    )

    parser.add_argument(
        "--use_ddp",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="DDP training True/False",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="World size/Number of GPUs in use for DDP training",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )

    parser.add_argument(
        "--dummy_plot", action="store_true", help="plot inter result for debug"
    )
    parser.add_argument(
        "--dummy_val", action="store_true", help="plot inter result for debug"
    )
    # transforms to apply
    parser.add_argument(
        "--transforms_train",
        type=str,
        nargs="+",
        default=["degrees", "scale"],
        choices=["translate", "flipud", "mixup", "degrees", "mosaic", "scale"],
        help="List of transforms to be applied",
    )

    return parser.parse_args()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    config = parse()
    solver = Solver(config)
    if config.mode == "train":
        solver.train()
        # Solver.save_output_yaml(config)
    elif config.mode == "val":
        config.save_txt = False
        results_val, _, _ = solver.test(
            config.val_dataset,
            config.weights,
            config.batch_size,
            config.img_size,
            config.conf_thres,
            config.iou_thres,
            False,
            config.single_cls,
            False,
            config.verbose,
            class_num=config.class_number,
        )
        # generate val_out.yml for debug view
        if config.arch != "mobilenetssd":
            solver.output.generate_val_out_yaml(
                config.val_dataset,
                config.weights,
                config.yaml_batch_size,
                config.img_size,
                config.conf_thres,
                config.iou_thres,
                False,
                config.single_cls,
                False,
                config.verbose,
                class_num=config.class_number,
                config=config,
            )
            solver.output.save_summary(config, 0, 0, float(results_val[2]))
        else:
            solver.output.save_summary(config, 0, 0, float(results_val))

    elif config.mode == "test":
        config.save_txt = False
        if config.arch != "mobilenetssd":
            solver.output.generate_val_out_yaml(
                config.test_dataset,
                config.weights,
                config.yaml_batch_size,
                config.img_size,
                config.conf_thres,
                config.iou_thres,
                False,
                config.single_cls,
                False,
                config.verbose,
                class_num=config.class_number,
                config=config,
            )
        else:
            solver.test(
                config.val_dataset,
                config.weights,
                config.batch_size,
                config.img_size,
                config.conf_thres,
                config.iou_thres,
                False,
                config.single_cls,
                False,
                config.verbose,
                class_num=config.class_number,
            )

        # solver.save_output_yaml(config)
    else:
        print("plz set mode between train or test or val")


# TODO:
