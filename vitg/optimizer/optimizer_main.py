import inspect

import torch.optim as py_optim

from . import optimizer_lib as custom_optim

"""
pytorch_optim_list \
    = [m for m in dir(py_optim)
       if inspect.isclass(getattr(py_optim, m)) and m[0] is not '_']

pytorch_optim_list.remove('Optimizer')

optimizer_param_dict = [
    {'name': 'lr', 'type': float, 'nargs': None, 'help': 'learning rate'},
    {'name': 'momentum', 'type': float, 'nargs': None, 'help': 'momentum'},
    {'name': 'weight_decay', 'type': float, 'nargs': None, 'help': 'weight_decay'},
    {'name': 'eps', 'type': float, 'nargs': None, 'help': 'term added to the denominator to improve numerical stability'}
]
"""


pytorch_optim_list = ["Adam", "AdamW", "SparseAdam", "Adamax", "SGD"]
optimizer_param_dict = [
    {
        "name": "lr",
        "type": float,
        "nargs": None,
        "default": 0.01,
        "help": "learning rate",
    },
    {
        "name": "momentum",
        "type": float,
        "nargs": None,
        "default": 0.9,
        "help": "momentum",
    },
    {
        "name": "weight_decay",
        "type": float,
        "nargs": None,
        "default": 0.0005,
        "help": "weight_decay",
    },
    {
        "name": "eps",
        "type": float,
        "nargs": None,
        "default": 1e-06,
        "help": "term added to the denominator to improve numerical stability",
    },
    {
        "name": "betas",
        "type": float,
        "nargs": 2,
        "default": [0.9, 0.999],
        "help": "betas value for Adam Optimizer",
    },
]


def make_optimizer(optimizer_type, params, optimizer_dict):
    """
    Creating the optimizer
    Arguments:
        optimizer_type: Type of Optimizer
        params: Model Parameters
        optimizer_dict: Full list of optimizer dictionary parameters
    return: Optimizer
    """
    # Pytorch Optimizer
    if optimizer_type in pytorch_optim_list:
        # Creating a optimizer parameter dictionary based on the optimizer type
        optim_type_dict = {}
        for k, v in optimizer_dict.items():
            optim_args = inspect.getfullargspec(getattr(py_optim, optimizer_type)).args
            if k in optim_args:
                optim_type_dict[k] = v
        return getattr(py_optim, optimizer_type)(params, **optim_type_dict)
    elif optimizer_type in custom_optim.__all__:
        # Creating a optimizer parameter dictionary based on the optimizer type
        optim_type_dict = {}
        for k, v in optimizer_dict.items():
            optim_args = inspect.getfullargspec(
                getattr(custom_optim, optimizer_type)
            ).args
            if k in optim_args:
                optim_type_dict[k] = v
        return getattr(custom_optim, optimizer_type)(params, **optim_type_dict)
    else:
        print(f"Optimizer function{optimizer_type}not available, Choose from:")
        print(pytorch_optim_list + custom_optim.__all__)


def get_optimizer_list():
    return pytorch_optim_list + custom_optim.__all__


def get_optimizer_param_dict():
    return optimizer_param_dict
