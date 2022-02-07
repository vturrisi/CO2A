import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributed as dist
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics import ConfusionMatrix as confusion_matrix


def accuracy_at_k(output, target, top_k=(1, 5)):
    """Computes the accuracy@k

    Args:
        output: output of the model
        target: real targets
        top_k (tuple, optional): @ values. Defaults to (1, 5).

    Returns:
        accuracy values at different k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_world_size():
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    return world_size


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def compute_entropy(x):
    """Computes entropy of input x."""
    epsilon = 1e-5
    H = -x * torch.log(x + epsilon)
    H = torch.sum(H, dim=1)
    return H


class EpochCheckpointer(Callback):
    """Checkpointer callback that saves models every frequency of epochs and at the end."""

    def __init__(self, args, logdir="trained_models", frequency=25):
        self.args = args
        self.frequency = frequency
        self.logdir = logdir

    def initial_setup(self, trainer):
        version = str(trainer.logger.version)
        if version is not None:
            self.path = os.path.join(self.logdir, version)
            self.ckpt_placeholder = f"{self.args.name}-{version}" + "-ep={}.ckpt"
        else:
            self.path = self.logdir
            self.ckpt_placeholder = f"{self.args.name}" + "-ep={}.ckpt"

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def save_args(self, trainer):
        if trainer.is_global_zero:
            args = vars(self.args)
            json_path = os.path.join(self.path, "args.json")
            json.dump(args, open(json_path, "w"))

    def save(self, trainer):
        epoch = trainer.current_epoch
        ckpt = self.ckpt_placeholder.format(epoch)
        trainer.save_checkpoint(os.path.join(self.path, ckpt))

    def on_train_start(self, trainer, pl_module):
        self.initial_setup(trainer)
        self.save_args(trainer)

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.frequency == 0 and epoch != 0:
            self.save(trainer)

    def on_train_end(self, trainer, pl_module):
        self.save(trainer)


class PseudoLabelDistribution(Callback):
    """Callback that computes pseudo-label distribution of the target dataset."""

    def __init__(self, args):
        self.args = args

    def on_train_epoch_start(self, trainer, module):
        # histograms
        self.hist_passed = []
        self.hist_filtered = []
        self.hist_total = []

        # distributions
        self.dist_per_pseudo_passed = []
        self.dist_per_pseudo_filtered = []
        self.dist_per_pseudo_total = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for gpu_outputs in outputs:
            for output in gpu_outputs:
                output = output["extra"]
                if "hist_passed" in output:
                    hist_passed = output["hist_passed"].cpu()
                    hist_filtered = output["hist_filtered"].cpu()
                    hist_total = output["hist_total"].cpu()

                    self.hist_passed.append(hist_passed)
                    self.hist_filtered.append(hist_filtered)
                    self.hist_total.append(hist_total)

                    dist_per_pseudo_passed = output["dist_per_pseudo_passed"]
                    dist_per_pseudo_filtered = output["dist_per_pseudo_filtered"]
                    dist_per_pseudo_total = output["dist_per_pseudo_total"]

                    self.dist_per_pseudo_passed.append(dist_per_pseudo_passed)
                    self.dist_per_pseudo_filtered.append(dist_per_pseudo_filtered)
                    self.dist_per_pseudo_total.append(dist_per_pseudo_total)

    def on_train_epoch_end(self, trainer, module, outputs):
        if trainer.is_global_zero:
            if len(self.hist_total):
                hist_total = torch.stack(self.hist_total)
                dist_total = hist_total.sum(0)
                ylim = (0, dist_total.max())

                for name, hist, dist_per_class in zip(
                    ["passed", "filtered", "total"],
                    [self.hist_passed, self.hist_filtered, self.hist_total],
                    [
                        self.dist_per_pseudo_passed,
                        self.dist_per_pseudo_filtered,
                        self.dist_per_pseudo_total,
                    ],
                ):
                    hist = torch.stack(hist)
                    dist = hist.sum(0)

                    # get distributions per class
                    sum_dist_per_class = defaultdict(list)
                    for v in dist_per_class:
                        for c, d in v.items():
                            if d is not None:
                                sum_dist_per_class[c].append(d.cpu())
                    temp = {c: torch.stack(d).sum(dim=0) for c, d in sum_dist_per_class.items()}
                    sum_dist_per_class = {c: None for c in range(module.num_classes)}
                    for c, v in temp.items():
                        sum_dist_per_class[c] = v

                    sns.set(rc={"figure.figsize": (30, 30), "font.size": 15})
                    sns.set(font_scale=2)
                    ax = sns.barplot(data=dist)
                    ax.set(ylim=ylim)

                    xticks = []
                    for c, d in sorted(sum_dist_per_class.items()):
                        if d is not None:
                            acc = round((d[c] / d.sum()).item(), 3)
                            h = round(compute_entropy((d / d.sum()).unsqueeze(0)).item(), 3)
                            d = str(d.int().tolist())
                        else:
                            acc = -1
                            h = -1
                        xticks.append(f"acc={acc},H={h},{d}")
                    ax.set_xticklabels(
                        xticks,
                        rotation=20,
                        horizontalalignment="right",
                        fontsize="medium",
                    )
                    plt.tight_layout()
                    if self.args.wandb:
                        wandb.log({f"pseudo_labels_dist_{name}": wandb.Image(ax)}, commit=False)
                        plt.close()


class ConfusionMatrix(Callback):
    """Callback that generates a confusion matrix."""

    def __init__(self, args):
        self.args = args

    def on_train_epoch_start(self, trainer, module):
        self.outputs_s = []
        self.targets_s = []

        self.outputs_t = []
        self.targets_t = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for gpu_outputs in outputs:
            for output in gpu_outputs:
                output = output["extra"]
                if "out_s" in output:
                    self.outputs_s.append(output["out_s"].cpu())
                    self.targets_s.append(output["y_source"].cpu())

                if "out_t" in output:
                    self.outputs_t.append(output["out_t"].cpu())
                    self.targets_t.append(output["y_target"].cpu())

    def on_train_epoch_end(self, trainer, module, outputs):
        if trainer.is_global_zero:
            for name, outputs, targets in zip(
                ["source", "target"],
                [self.outputs_s, self.outputs_t],
                [self.targets_s, self.targets_t],
            ):
                if len(outputs):
                    outputs = torch.cat(outputs)
                    targets = torch.cat(targets)

                    preds = outputs.float().max(dim=1)[1]

                    cm = confusion_matrix(module.num_classes)(preds, targets).cpu()
                    sns.set(rc={"figure.figsize": (30, 30), "font.size": 15})
                    sns.set(font_scale=2)
                    ax = sns.heatmap(data=cm, annot=True, cmap="OrRd")
                    values = list(range(cm.size(0)))
                    ax.set_xticklabels(values, rotation=45, fontsize="large")
                    ax.set_yticklabels(values, rotation=90, fontsize="large")
                    plt.tight_layout()
                    if self.args.wandb:
                        wandb.log({f"train_{name}_cm": wandb.Image(ax)}, commit=False)
                        plt.close()

    def on_validation_epoch_start(self, trainer, module):
        self.outputs = []
        self.targets = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.outputs.append(outputs["outputs"].cpu())
        self.targets.append(outputs["targets"].cpu())

    def on_validation_epoch_end(self, trainer, module):
        if trainer.is_global_zero:
            self.outputs = torch.cat(self.outputs)
            self.targets = torch.cat(self.targets)

            preds = self.outputs.float().max(dim=1)[1]
            targets = self.targets

            cm = confusion_matrix(module.num_classes)(preds, targets).cpu()
            if cm.size():
                sns.set(rc={"figure.figsize": (30, 30), "font.size": 15})
                sns.set(font_scale=2)
                ax = sns.heatmap(data=cm, annot=True, cmap="OrRd")
                values = list(range(cm.size(0)))
                ax.set_xticklabels(values, rotation=45, fontsize="large")
                ax.set_yticklabels(values, rotation=90, fontsize="large")
                plt.tight_layout()
                if self.args.wandb:
                    wandb.log({"val_cm": wandb.Image(ax)}, commit=False)
                    plt.close()
