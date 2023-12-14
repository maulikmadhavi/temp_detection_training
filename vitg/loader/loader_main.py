from tqdm import tqdm

from vitg.utils.datasets import (
    create_dataloader,
    create_test_dataloader,
    create_dataloader_noletterbox,
)


def get_test_loader(config, hyp):
    (
        testloader_noaug,
        trainloader_noaug,
        testloader_train_yml,
        testloader_val_yml,
    ) = (None, None, None, None)
    batch_size = config.batch_size
    gs = config.gs
    imgsz_test = config.imgsz_test    
    if config.arch == "mobilenetssd":
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader_noaug = create_dataloader_noletterbox(
            config.val_dataset,
            300,
            batch_size,
            0,
            config,
            hyp=None,
            augment=False,
            pad=0,
            rect=False,
        )[0]
        trainloader_noaug = create_dataloader_noletterbox(
            config.train_dataset,
            300,
            batch_size,
            0,
            config,
            hyp=None,
            augment=False,
            pad=0,
            rect=False,
        )[0]
    elif config.arch == "yolov8":
        testloader_noaug = create_dataloader_noletterbox(
            config.val_dataset,
            imgsz_test,
            batch_size,
            0,
            config,
            hyp=None,
            augment=False,
            pad=0,
            rect=False,
        )[0]
    else:
        testloader_noaug = create_dataloader(
            config.val_dataset,
            imgsz_test,
            1,
            gs,
            config,
            hyp=hyp,
            augment=False,
            rect=True,
            local_rank=-1,
            world_size=config.world_size,
        )[0]
        # train_out.yml
    testloader_train_yml = create_test_dataloader(
        config.train_dataset,
        imgsz_test,
        1,
        1,
        config,
        hyp=hyp,
        augment=False,
        rect=True,
        local_rank=-1,
        world_size=config.world_size,
    )[0]

    # val_out.yml
    testloader_val_yml = create_test_dataloader(
        config.val_dataset,
        imgsz_test,
        1,
        1,
        config,
        hyp=hyp,
        augment=False,
        rect=True,
        local_rank=-1,
        world_size=config.world_size,
    )[0]

    return (
        testloader_noaug,
        trainloader_noaug,
        testloader_train_yml,
        testloader_val_yml,
    )


def get_intertrain_loader(config, hyp):
    batch_size = config.batch_size
    gs = config.gs
    imgsz_test = config.imgsz_test    
    return (
        create_dataloader_noletterbox(
            config.train_dataset,
            imgsz_test,
            batch_size,
            0,
            config,
            hyp=None,
            augment=False,
            pad=0,
            rect=False,
        )[0]
        if config.arch == "yolov8"
        else create_dataloader(
            config.train_dataset,
            imgsz_test,
            batch_size,
            gs,
            config,
            hyp=hyp,
            augment=False,
            rect=True,
            local_rank=-1,
            world_size=config.world_size,
        )[0]
    )


def get_train_loader(config, hyp):
    batch_size = config.batch_size
    use_rec_train = not hyp["mosaic"]
    rank = config.rank
    gs = config.gs
    imgsz = config.imgsz
    if config.arch == "mobilenetssd" or config.arch == "yolov8":
        use_rect = True
        dataloader, dataset = create_dataloader_noletterbox(
            config.train_dataset,
            300 if config.arch == "mobilenetssd" else imgsz,
            batch_size,
            gs,
            config,
            hyp=hyp,
            augment=True,
            pad=0,
            rect=True,
        )
    else:
        # use_rect = False
        dataloader, dataset = create_dataloader(
            config.train_dataset,
            imgsz,
            batch_size,
            gs,
            config,
            hyp=hyp,
            augment=True,
            rect=use_rec_train,
            local_rank=rank,
            world_size=config.world_size,
        )
    return dataloader
