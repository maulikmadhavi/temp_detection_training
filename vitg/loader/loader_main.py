from vitg.utils.datasets import (
    create_dataloader,
    create_dataloader_noletterbox,
    create_test_dataloader,
)


def get_test_loader(config, hyp, batch_size, gs, imgsz_test):
    (
        testloader_noaug,
        trainloader_noaug,
        testloader_train_yml,
        testloader_val_yml,
    ) = (None, None, None, None)
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
            cache=False,
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
            cache=False,
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
            cache=False,
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
            cache=False,
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
        cache=False,
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
        cache=False,
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


def get_intertrain_loader(config, hyp, batch_size, gs, imgsz_test):
    return (
        create_dataloader_noletterbox(
            config.train_dataset,
            imgsz_test,
            batch_size,
            0,
            config,
            hyp=None,
            augment=False,
            cache=False,
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
            cache=False,
            rect=True,
            local_rank=-1,
            world_size=config.world_size,
        )[0]
    )


def get_train_loader(config, hyp, batch_size, rank, gs, imgsz, use_rec_train):
    if config.arch == "mobilenetssd":
        dataloader, _ = create_dataloader_noletterbox(
            config.train_dataset,
            300,
            batch_size,
            gs,
            config,
            hyp=hyp,
            augment=True,
            cache=False,
            pad=0,
            rect=True,
        )
    elif config.arch == "yolov8":
        dataloader, _ = create_dataloader_noletterbox(
            config.train_dataset,
            imgsz,
            batch_size,
            gs,
            config,
            hyp=hyp,
            augment=True,
            cache=False,
            pad=0,
            rect=True,
        )
    else:
        dataloader, _ = create_dataloader(
            config.train_dataset,
            imgsz,
            batch_size,
            gs,
            config,
            hyp=hyp,
            augment=True,
            cache=False,
            rect=use_rec_train,
            local_rank=rank,
            world_size=config.world_size,
        )
    return dataloader
