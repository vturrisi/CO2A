import argparse
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.dataloader import prepare_data_source_only
from src.i3d import InceptionI3d, load_i3d_imagenet_pretrained
from src.utils import ConfusionMatrix, EpochCheckpointer, PseudoLabelDistribution
from src.video_model import PretrainVideoModel


def parse_args():
    SUP_OPT = ["sgd", "adam"]
    SUP_SCHED = ["reduce", "cosine", "step", "exponential", "none"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", type=str)
    parser.add_argument("--val_dataset", type=str)

    # optimizer
    parser.add_argument("--optimizer", default="sgd", choices=SUP_OPT)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    # scheduler
    parser.add_argument("--scheduler", choices=SUP_SCHED, default="reduce")
    parser.add_argument(
        "-lr_decay_steps",
        "--lr_decay_steps",
        default=[200, 300, 350],
        type=int,
        nargs="+",
    )

    # general settings
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=8)

    # training settings
    parser.add_argument("--resume_training_from", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+")

    # extra model stuff
    parser.add_argument("--bottleneck_size", type=int, default=256)
    parser.add_argument("--projection_size", type=int, default=128)
    parser.add_argument(
        "--aggregation", choices=["avg", "lstm", "lstm_weights", "mlp", "mlp_weights"]
    )
    parser.add_argument("--video_dropout", type=float, default=0.5)

    # I3D pretraining
    parser.add_argument("--imagenet_pretrained", action="store_true")

    # data stuff
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--n_clips", type=int, default=4)
    parser.add_argument("--source_augmentations", default=[], nargs="+")

    parser.add_argument("--source_patch_aug_folder", default=None)
    parser.add_argument(
        "--source_patch_aug_mode", default=None, choices=["diff_video", "same_video"]
    )
    parser.add_argument("--patch_size", default=64, type=int)

    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--wandb", action="store_true")

    # backend (for docker?)
    parser.add_argument("--distributed_backend", default="ddp", choices=["ddp"])

    # other
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--add_bn", action="store_true")
    parser.add_argument("--layers_ca", type=int, default=1)
    parser.add_argument("--add_bn_ca", action="store_true")
    parser.add_argument("--third_projection", action="store_true")
    parser.add_argument("--oracle", action="store_true")

    args = parser.parse_args()

    # find number of classes
    args.num_classes = len(set(os.listdir(args.source_dataset)))

    # add momentum if sgd
    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    # assert settings for source patch augmentation
    if "patch" in args.source_augmentations:
        assert args.source_patch_aug_folder is not None
        assert args.source_patch_aug_mode is not None
    args.source_augmentations_params = {
        "patch": {
            "folder": args.source_patch_aug_folder,
            "mode": args.source_patch_aug_mode,
            "size": args.patch_size,
        }
    }

    args.no_task_block = False
    args.temperature = 1
    args.ce_loss_weight = 1
    args.contrastive_loss_weight = 0

    return args


def main():
    args = parse_args()

    model = InceptionI3d()
    if args.imagenet_pretrained:
        ckp = load_i3d_imagenet_pretrained()
        model.load_state_dict(ckp)

    model = PretrainVideoModel(model, args.num_classes, args)

    # dataloader
    source_loader, val_loader = prepare_data_source_only(
        args.source_dataset,
        args.val_dataset,
        n_frames=args.n_frames,
        n_clips=args.n_clips,
        frame_size=args.frame_size,
        augmentations=args.source_augmentations,
        augmentations_params=args.source_augmentations_params,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        divide_mixamo=True,
    )

    # epoch checkpointer
    checkpointer = EpochCheckpointer(args, frequency=25)
    pseudo_label_stats = PseudoLabelDistribution(args)
    cm = ConfusionMatrix(args)
    callbacks = [checkpointer, pseudo_label_stats, cm]

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(name=args.name, project=args.project)
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=[*args.gpus],
        logger=wandb_logger if args.wandb else None,
        distributed_backend=args.distributed_backend,
        precision=32 if args.aggregation in ["lstm", "lstm_weights"] else 16,
        sync_batchnorm=True,
        resume_from_checkpoint=args.resume_training_from,
        callbacks=callbacks,
    )
    trainer.fit(model, source_loader, val_loader)


if __name__ == "__main__":
    seed_everything(5)

    main()
