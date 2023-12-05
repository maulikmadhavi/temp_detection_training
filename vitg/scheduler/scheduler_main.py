import inspect

import torch.optim.lr_scheduler as py_sched

from . import scheduler_lib as custom_sched

"""
pytorch_sched_list \
    = [m[0] for m in inspect.getmembers(py_sched, inspect.isclass)
       if m[1].__module__ == py_sched.__name__ and m[0][0] is not '_']

scheduler_param_dict = [
    {'name': 'milestones', 'type': int, 'nargs': '+', 'help': 'List of epoch indices. Must be increasing.'},
    {'name': 'gamma', 'type': float, 'nargs': None, 'help': 'Multiplicative factor of learning rate decay'},
    {'name': 'lr_lambda', 'type': list, 'nargs': '+', 'help': 'A function which computes a multiplicative factor '
                                                              'given an integer parameter epoch, or a list of such '
                                                              'functions, one for each group in optimizer.param_groups'},
    {'name': 'step_size', 'type': int, 'nargs': None, 'help': 'Period of learning rate decay'},
    {'name': 'T_max', 'type': int, 'nargs': None, 'help': 'Maximum number of iterations'},
    {'name': 'base_lr', 'type': float, 'nargs': '+',
     'help': 'Initial learning rate which is the lower boundary in the cycle for each parameter group'},
    {'name': 'max_lr', 'type': float, 'nargs': '+',
     'help': 'Upper learning rate boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_lr - base_lr).The lr at any cycle is the sum of base_lr and some scaling of the amplitude; therefore max_lr may not actually be reached depending on scaling function.'},
    {'name': 'T_0', 'type': int, 'nargs': None, 'help': 'Number of iterations for the first restart'}
]
"""

pytorch_sched_list = ["StepLR", "MultiStepLR"]
scheduler_param_dict = [
    {
        "name": "milestones",
        "type": int,
        "nargs": "+",
        "default": [50, 75, 100],
        "help": "List of epoch indices. Must be increasing.",
    },
    {
        "name": "gamma",
        "type": float,
        "nargs": None,
        "default": 0.1,
        "help": "Multiplicative factor of learning rate decay",
    },
    {
        "name": "step_size",
        "type": int,
        "nargs": None,
        "default": 50,
        "help": "Period of learning rate decay",
    },
    {
        "name": "first_cycle_steps",
        "type": int,
        "nargs": None,
        "default": 100,
        "help": "Number of epochs for the first restart in warup scheduler",
    },
    {
        "name": "warmup_step",
        "type": int,
        "nargs": None,
        "default": 5,
        "help": "Linear warmup step size",
    },
    {
        "name": "cycle_mult",
        "type": float,
        "nargs": None,
        "default": 1.0,
        "help": "Cycle steps magnification",
    },
    {
        "name": "pct_start",
        "type": float,
        "nargs": None,
        "default": 0.3,
        "help": "The percentage of the cycle (in number of steps) spent increasing the learning rate.",
    },
]


def make_scheduler(scheduler_type, optimizer, scheduler_dict):
    """
    Creating the scheduler
    Arguments:
        scheduler_type: Type of scheduler
        optimizer: Optimizer
        scheduler_dict: Full list of scheduler dictionary parameters
    return: scheduler
    """
    # Pytorch Scheduler
    if scheduler_type in pytorch_sched_list:
        # Creating a scheduler parameter dictionary based on the scheduler type
        sched_type_dict = {}
        for k, v in scheduler_dict.items():
            sched_args = inspect.getfullargspec(getattr(py_sched, scheduler_type)).args
            if k in sched_args:
                sched_type_dict[k] = v
        scheduler_func = getattr(py_sched, scheduler_type)(optimizer, **sched_type_dict)
        is_custom = False
        return scheduler_func, is_custom
    elif scheduler_type in custom_sched.__all__:
        # Creating a scheduler parameter dictionary based on the scheduler type
        sched_type_dict = {}
        for k, v in scheduler_dict.items():
            sched_args = inspect.getfullargspec(
                getattr(custom_sched, scheduler_type)
            ).args
            if k in sched_args:
                sched_type_dict[k] = v
        scheduler_func = getattr(custom_sched, scheduler_type)(
            optimizer, **sched_type_dict
        )
        is_custom = True
        return scheduler_func, is_custom
    else:
        print(f"Scheduler function{scheduler_type}not available, Choose from:")
        print(pytorch_sched_list)


def get_scheduler_list():
    return pytorch_sched_list + custom_sched.__all__


def get_scheduler_param_dict():
    return scheduler_param_dict
