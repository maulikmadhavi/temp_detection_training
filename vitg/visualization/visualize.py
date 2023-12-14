import os

from torch.utils.tensorboard import SummaryWriter


class Visualize:
    """
    Visualization for Training: Currently supports Tensorboard
    """

    def __init__(self, visualize_type="tensorboard"):
        self.writer = SummaryWriter() if visualize_type == "tensorboard" else None

    def update(self, run, results, mloss):
        if self.writer:
            tags = [
                "train/giou_loss",
                "train/obj_loss",
                "train/cls_loss",
                "metrics/precision",
                "metrics/recall",
                "metrics/mAP_0.5",
                "metrics/mAP_0.5:0.95",
                "val/giou_loss",
                "val/obj_loss",
                "val/cls_loss",
            ]
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                self.writer.add_scalar(tag, x, run)

    def save_weight(self):
        wdir = os.path.join(
            self.writer.log_dir, "evolve", "weights"
        )  # weights directory
        os.makedirs(wdir, exist_ok=True)
