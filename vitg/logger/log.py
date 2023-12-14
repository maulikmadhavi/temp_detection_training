import logging
import os
import sys
import torch


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass

        return no_op


def get_logger(log_path, is_rank0=True):
    """Get the program logger.
    This is used with DDP as we need to use logging for rank=0 and
    bypass all operations with other ranks.
    Args:
        log_path (str):  The log filename.
        is_rank0 (boolean): If True, create the normal logger; If False,
        create the null logger, which is useful in DDP training. Default is True.
    """
    if is_rank0:
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))

        logger = logging.getLogger("my_logger")
        logging.basicConfig(level=logging.DEBUG)
        basic_formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

        logger.setLevel(level=logging.DEBUG)

        # StreamHandler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.DEBUG)
        logger.addHandler(stream_handler)

        # FileHandler
        mode = "a+"
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(basic_formatter)  # To make same as terminal
        logger.addHandler(file_handler)
    else:
        logger = NoOp()

    return logger


class ShowProgressBar:
    def __init__(self, arch, device):
        if arch != "mobilenetssd":
            print(
                ("\n" + "%10s" * 8)
                % (
                    "Epoch",
                    "gpu_mem",
                    "GIoU",
                    "obj",
                    "cls",
                    "total",
                    "targets",
                    "img_size",
                )
            )
        else:
            print(
                ("\n" + "%10s" * 5)
                % (
                    "Epoch",
                    "gpu_mem",
                    "total",
                    "targets",
                    "img_size",
                )
            )
        self.arch = arch
        self.mloss = torch.zeros(4, device=device)  # mean losses
        self.i = 0

    def __call__(self, epoch_epochs, loss_list, data):
        epoch, epochs = epoch_epochs
        imgs, targets = data
        loss, loss_items = loss_list

        mem = self.get_gpu_mem()
        if self.arch == "mobilenetssd":
            return ("%10s" * 2 + "%10.3g" * 1 + "%10.3g" * 2) % (
                f"{epoch}/{epochs - 1}",
                f"{mem}",
                loss.item(),
                len(targets),
                imgs.shape[-1],
            )
        self.mloss = (self.mloss * self.i + loss_items) / (
            self.i + 1
        )  # update mean losses

        losses = self.mloss.detach().cpu().numpy().tolist()
        return ("%10s" * 2 + "%10.3g" * 4 + "%10.3g" * 2) % (
            f"{epoch}/{epochs - 1}",
            f"{mem}",
            losses[0],
            losses[1],
            losses[2],
            losses[3],
            len(targets),
            imgs.shape[-1],
        )

    def get_gpu_mem(self):
        return "%.3gG" % (
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        )  # (GB)
