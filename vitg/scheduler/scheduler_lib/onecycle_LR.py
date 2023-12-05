from functools import partial

import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler


class OneCyclePolicy(_LRScheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, pct_start: float, last_epoch: int = -1
    ) -> None:
        self.sch = partial(lr_scheduler.OneCycleLR, optimizer, pct_start=pct_start)
        # Partial function to be completed by the total number of steps inside solver.py using config

    def get_lr(self) -> float:
        return self.sch.get_lr()
