import argparse
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.dataloader import prepare_data, prepare_data_source_only
from src.i3d import InceptionI3d, load_i3d_imagenet_pretrained
from src.utils import ConfusionMatrix, EpochCheckpointer, PseudoLabelDistribution
from src.video_model import VideoModel
from pytorch_lightning.plugins import DDPPlugin

torch.backends.cudnn.benchmark = True


def parse_args():
    SUP_OPT = ["sgd", "adam"]
    SUP_SCHED = ["reduce", "cosine", "step", "exponential", "none"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", type=str)
    parser.add_argument("--target_dataset", type=str)
    parser.add_argument("--val_dataset", type=str)

    # optimizer
    parser.add_argument("--optimizer", default="sgd", choices=SUP_OPT)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    # scheduler
    parser.add_argument("--scheduler", choices=SUP_SCHED, default="reduce")
    parser.add_argument("--lr_steps", type=int, nargs="+")

    # general settings
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=8)

    # training settings
    parser.add_argument("--resume_training_from", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+")

    # contrastive stuff
    parser.add_argument("--temperature", type=float, default=0.2)
    # loss weights
    parser.add_argument("--ce_loss_weight", type=float, default=1)
    parser.add_argument("--ce_loss_target_weight", type=float, default=0)
    parser.add_argument("--nce_loss_target_aug_based_weight", type=float, default=0)
    parser.add_argument("--nce_loss_target_clip_aug_based_weight", type=float, default=0)
    parser.add_argument("--nce_loss_source_label_based_weight", type=float, default=0)
    parser.add_argument("--nce_loss_target_label_based_weight", type=float, default=0)
    parser.add_argument("--nce_loss_inter_domain_weight", type=float, default=0)

    # consistency stuff
    parser.add_argument("--consistency_loss_weight", type=float, default=0)
    parser.add_argument("--consistency_threshold", type=float, default=0.5)
    parser.add_argument("--complete_nce_weight", type=float, default=0)
    # use supervised labels instead of pseudo
    parser.add_argument("--supervised_labels", action="store_true")
    # factor to filter out instances
    parser.add_argument("--selection_factor", type=int, default=6)

    # extra model stuff
    parser.add_argument("--bottleneck_size", type=int, default=256)
    parser.add_argument("--projection_size", type=int, default=128)

    # debug stuff for the heads
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--add_bn", action="store_true")
    parser.add_argument("--layers_ca", type=int, default=1)
    parser.add_argument("--add_bn_ca", action="store_true")
    parser.add_argument("--third_projection", action="store_true")
    parser.add_argument("--oracle", action="store_true")

    parser.add_argument(
        "--aggregation",
        choices=[
            "avg",
            "lstm",
            "lstm_weights",
            "mlp",
            "mlp_weights",
        ],
    )
    parser.add_argument("--video_dropout", type=float, default=0)

    # I3D pretraining
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--imagenet_pretrained", action="store_true")
    parser.add_argument("--mixamo_pretrained", action="store_true")
    parser.add_argument("--mixamo14_pretrained", action="store_true")
    parser.add_argument("--mixamo_pretrained_final", action="store_true")

    # data stuff
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--n_clips", type=int, default=4)
    parser.add_argument("--source_augmentations", default=[], nargs="+")
    parser.add_argument("--target_augmentations", default=[], nargs="+")
    parser.add_argument("--target_2_augs", action="store_true")

    # ablation (?)
    parser.add_argument("--source_only", action="store_true")
    parser.add_argument("--no_task_block", action="store_true")
    parser.add_argument("--source_source", action="store_true")
    parser.add_argument("--target_target", action="store_true")
    parser.add_argument("--source_target", action="store_true")

    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--wandb", action="store_true")

    # backend (for docker?)
    parser.add_argument("--distributed_backend", default="ddp", choices=["dp", "ddp"])

    args = parser.parse_args()

    # find number of classes
    args.num_classes = len(set(os.listdir(args.source_dataset)))

    # only one type of pretraining is allowed
    assert (
        (args.pretrained and not args.imagenet_pretrained)
        or (not args.pretrained and args.imagenet_pretrained)
        or (not args.pretrained and not args.imagenet_pretrained)
    )

    return args


def main():
    args = parse_args()

    # load backbone and weights
    model = InceptionI3d()
    if args.pretrained:
        ckp = torch.load("../pretrained/rgb_imagenet.pt", map_location="cpu")
        model.load_state_dict(ckp, strict=False)
    elif args.imagenet_pretrained:
        ckp = load_i3d_imagenet_pretrained()
        model.load_state_dict(ckp)
    elif args.mixamo14_pretrained:
        ckp = torch.load("../pretrained/mixamo_pretrained.pt", map_location="cpu")
        state_dict = {}
        for k, v in ckp["state_dict"].items():
            if k.startswith("base."):
                state_dict[k.replace("base.", "")] = v
        model.load_state_dict(state_dict)

    model = VideoModel(model, args.num_classes, args)

    # dataloader
    if args.source_only:
        source_loader, val_loader = prepare_data_source_only(
            args.source_dataset,
            args.val_dataset,
            n_frames=args.n_frames,
            n_clips=args.n_clips,
            frame_size=args.frame_size,
            augmentations=args.source_augmentations,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        source_loader, val_loader = prepare_data(
            args.source_dataset,
            args.target_dataset,
            args.val_dataset,
            n_frames=args.n_frames,
            n_clips=args.n_clips,
            frame_size=args.frame_size,
            source_augmentations=args.source_augmentations,
            target_augmentations=args.target_augmentations,
            target_2_augs=args.target_2_augs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
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

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=[*args.gpus],
        logger=wandb_logger if args.wandb else None,
        distributed_backend=args.distributed_backend,
        precision=32 if args.aggregation in ["lstm", "lstm_weights"] else 16,
        sync_batchnorm=True,
        resume_from_checkpoint=args.resume_training_from,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, source_loader, val_loader)


if __name__ == "__main__":
    seed_everything(5)

    main()
