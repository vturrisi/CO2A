import math

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.metrics import mean

from .loss import AugBasedNCE, FilteredLabelBasedNCE, InterDomainBasedNCE, LabelBasedNCE
from .utils import GatherLayer, accuracy_at_k, compute_entropy


class Block(nn.Module):
    def __init__(self, in_size, out_size, layers=1, add_bn=False, dropout=0.0):
        super().__init__()

        self.mlp = []
        self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(nn.Linear(in_size, out_size))
        if add_bn:
            self.mlp.append(nn.BatchNorm1d(out_size))
        self.mlp.append(nn.ReLU(inplace=True))

        for i in range(layers - 1):
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(nn.Linear(out_size, out_size))
            if add_bn:
                self.mlp.append(nn.BatchNorm1d(out_size))
            self.mlp.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        x = self.mlp(x)
        return x


class AvgPoolAggregation(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=1)


class MLPAggregation(nn.Module):
    def __init__(self, feature_size, n_clips, layers=1, add_bn=False, dropout=0.0):
        super().__init__()

        self.n_clips = n_clips

        self.mlp = []
        self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(nn.Linear(self.n_clips * feature_size, feature_size))
        if add_bn:
            self.mlp.append(nn.BatchNorm1d(feature_size))
        self.mlp.append(nn.ReLU(inplace=True))

        for i in range(layers - 2):
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(nn.Linear(feature_size, feature_size))
            if add_bn:
                self.mlp.append(nn.BatchNorm1d(feature_size))
            self.mlp.append(nn.ReLU(inplace=True))

        self.mlp.append(nn.Linear(feature_size, feature_size))

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)
        x = self.mlp(x)
        return x


class MLPAggregationWeights(nn.Module):
    def __init__(self, feature_size, n_clips, layers=1, add_bn=False, dropout=0.0):
        super().__init__()

        self.n_clips = n_clips
        self.mlp = []
        self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(nn.Linear(self.n_clips * feature_size, feature_size))
        if add_bn:
            self.mlp.append(nn.BatchNorm1d(feature_size))
        self.mlp.append(nn.ReLU(inplace=True))

        for i in range(layers - 2):
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(nn.Linear(feature_size, feature_size))
            if add_bn:
                self.mlp.append(nn.BatchNorm1d(feature_size))
            self.mlp.append(nn.ReLU(inplace=True))

        self.mlp.append(nn.Linear(feature_size, n_clips))
        self.mlp.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        b = x.size(0)
        x_unrolled = x.view(b, -1)
        weights = self.mlp(x_unrolled)
        x = x * weights.unsqueeze(2)
        return torch.mean(x, dim=1)


class LSTMAggregation(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size=feature_size, hidden_size=feature_size, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return x

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


class LSTMAggregationWeights(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size=feature_size, hidden_size=feature_size, batch_first=True)
        self.linear = nn.Linear(feature_size, 4)

    def forward(self, x):
        h, _ = self.rnn(x)
        h = h[:, -1, :]
        w = F.softmax(self.linear(h), dim=1)
        x = x * w.unsqueeze(2)
        x = torch.mean(x, dim=1)
        return x

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=4):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class VideoModel(pl.LightningModule):
    def __init__(
        self,
        base,
        num_classes,
        args,
    ):
        super().__init__()

        # define base model
        self.base = base
        self.num_classes = num_classes
        self.args = args
        with torch.no_grad():
            x = torch.zeros((2, 3, 16, args.frame_size, args.frame_size))
            out = self.base(x)

        self.repr_size = out.size(1)
        self.bottleneck_size = args.bottleneck_size
        self.aggregation = args.aggregation
        self.proj_size = args.projection_size
        self.n_clips = args.n_clips

        self.aug_based_contrastive = AugBasedNCE(temperature=args.temperature)
        self.label_based_contrastive = LabelBasedNCE(temperature=args.temperature)
        self.filtered_label_based_contrastive = FilteredLabelBasedNCE(temperature=args.temperature)
        self.inter_domain_based_contrastive = InterDomainBasedNCE(temperature=args.temperature)

        # define the rest of the model
        if self.aggregation == "avg":
            self.clip_aggregation = AvgPoolAggregation()
        elif self.aggregation == "lstm":
            self.clip_aggregation = LSTMAggregation(self.repr_size)
        elif self.aggregation == "lstm_weights":
            self.clip_aggregation = LSTMAggregationWeights(self.repr_size)
        elif self.aggregation == "mlp":
            self.clip_aggregation = MLPAggregation(
                self.repr_size,
                self.n_clips,
                layers=args.layers_ca,
                add_bn=args.add_bn_ca,
            )
        elif self.aggregation == "mlp_weights":
            self.clip_aggregation = MLPAggregationWeights(
                self.repr_size,
                self.n_clips,
                layers=args.layers_ca,
                add_bn=args.add_bn_ca,
            )
        else:
            raise ValueError(f"Clip aggregation {self.aggregation} not supported.")

        # classifier head
        self.head_class = Block(
            self.repr_size,
            self.bottleneck_size,
            layers=args.layers,
            dropout=args.video_dropout,
            add_bn=args.add_bn,
        )
        self.fc = nn.Linear(self.bottleneck_size, num_classes)

        # contrastive head
        self.head_cont = Block(
            self.repr_size,
            self.bottleneck_size,
            layers=args.layers,
            dropout=args.video_dropout,
            add_bn=args.add_bn,
        )

        # projection heads
        self.projection_head_idc = nn.Sequential(
            nn.Linear(self.repr_size, self.repr_size),
            nn.ReLU(),
            nn.Linear(self.repr_size, self.proj_size),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.bottleneck_size, self.bottleneck_size),
            nn.ReLU(),
            nn.Linear(self.bottleneck_size, self.proj_size),
        )
        self.clip_projection_head = nn.Sequential(
            nn.Linear(self.repr_size, self.repr_size),
            nn.ReLU(),
            nn.Linear(self.repr_size, self.proj_size),
        )

    def forward(self, x, return_features=False):
        if self.aggregation in ["lstm", "lstm_weights"]:
            self.clip_aggregation.flatten_parameters()

        b, n_clips, n_frames, c, h, w = x.size()

        # consider clip as extra video in the batch
        clip_features = x.view(b * n_clips, n_frames, c, h, w)

        # apply I3D
        clip_features = self.base(clip_features)

        clip_projected_features = self.clip_projection_head(clip_features.view(b * n_clips, -1))

        # divide features per clip
        clip_features = clip_features.view(b, n_clips, -1)

        # aggregate clips
        video_features = self.clip_aggregation(clip_features)

        # classifier head
        out_head_class = self.head_class(video_features)
        out = self.fc(out_head_class)

        # contrastive head
        out_head_con = self.head_cont(video_features)
        projected_features = self.projection_head(out_head_con)

        # contrastive projection idc
        projected_features_idc = self.projection_head_idc(video_features)

        if return_features:
            return (
                out,
                out_head_con,
                projected_features,
                projected_features_idc,
                clip_projected_features,
            )
        return out

    def configure_optimizers(self):
        # select optimizer
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD
            extra_optimizer_args = {"momentum": 0.9}
        else:
            optimizer = torch.optim.Adam
            extra_optimizer_args = {}

        optimizer = optimizer(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            **extra_optimizer_args,
        )

        if self.args.scheduler == "none":
            return optimizer
        else:
            if self.args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)
            elif self.args.scheduler == "reduce":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            elif self.args.scheduler == "step":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.lr_steps)
            elif self.args.scheduler == "exponential":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, self.args.weight_decay
                )
            return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        args = self.args

        # get data according to training mode
        if args.source_only:
            X_source, y_source = batch
            X = X_source
        elif args.target_2_augs:
            (
                source_index,
                X_source,
                y_source,
                target_index,
                X_target_1,
                X_target_2,
                y_target,
            ) = batch
            X = torch.cat((X_source, X_target_1, X_target_2), dim=0)
        else:
            source_index, X_source, y_source, target_index, X_target, y_target = batch
            X = torch.cat((X_source, X_target), dim=0)

        (
            out,
            out_head_con,
            projected_features,
            projected_features_idc,
            clip_projected_features,
        ) = self(X, return_features=True)

        if args.source_only:
            out_s = out
            out_con_s = out_head_con
            z_s = projected_features
            z_idc_s = projected_features_idc
            clip_z_s = clip_projected_features
        elif args.target_2_augs:
            b_s = y_source.size(0)

            out_s = out[:b_s]
            out_t = out[b_s:]
            out_t1, out_t2 = out_t.chunk(2)

            out_con_s = out_head_con[:b_s]
            out_con_t = out_head_con[b_s:]
            out_con_t1, out_con_t2 = out_con_t.chunk(2)

            z_s = projected_features[:b_s]
            z_t = projected_features[b_s:]
            z_t1, z_t2 = z_t.chunk(2)

            z_idc_s = projected_features_idc[:b_s]
            z_idc_t = projected_features_idc[b_s:]
            z_idc_t1, z_idc_t2 = z_t.chunk(2)

            clip_z_s = clip_projected_features[: b_s * args.n_clips]
            clip_z_t = clip_projected_features[b_s * args.n_clips :]
            clip_z_t1, clip_z_t2 = clip_z_t.chunk(2)

            y_target = y_target.repeat(2)
        else:
            out_s, out_t = out.chunk(2)
            out_con_s, out_con_t = out_head_con.chunk(2)
            z_s, z_t = projected_features.chunk(2)
            z_idc_s, z_idc_t = projected_features_idc.chunk(2)
            clip_z_s, clip_z_t = clip_projected_features.chunk(2)

        if args.source_only:
            loss = ce_loss = F.cross_entropy(out, y_source)
        else:
            # cross entropy loss for source and target
            ce_loss = F.cross_entropy(out_s, y_source)
            selection_target = (
                compute_entropy(F.softmax(out_t, dim=1))
                < math.log(self.num_classes) / self.args.selection_factor
            )
            y_target_est = out_t.max(dim=1)[1]
            ce_loss_target = torch.tensor(0.0, device=self.device)
            if torch.any(selection_target):
                ce_loss_target = F.cross_entropy(
                    out_t[selection_target], y_target_est[selection_target]
                )

            # gather variables across devices
            # source logits
            out_s = torch.cat(GatherLayer.apply(out_s), dim=0)
            # source contrastive features
            out_con_s = torch.cat(GatherLayer.apply(out_con_s), dim=0)

            # source projection head features
            z_s = torch.cat(GatherLayer.apply(z_s), dim=0)
            z_idc_s = torch.cat(GatherLayer.apply(z_idc_s), dim=0)
            clip_z_s = torch.cat(GatherLayer.apply(clip_z_s), dim=0)

            if args.target_2_augs:
                # target logits
                out_t1 = torch.cat(GatherLayer.apply(out_t1), dim=0)
                out_t2 = torch.cat(GatherLayer.apply(out_t2), dim=0)
                out_t = torch.cat((out_t1, out_t2))

                # target contrastive features
                out_con_t1 = torch.cat(GatherLayer.apply(out_con_t1), dim=0)
                out_con_t2 = torch.cat(GatherLayer.apply(out_con_t2), dim=0)
                out_con_t = torch.cat((out_con_t1, out_con_t2))

                # target projection head features
                z_t1 = torch.cat(GatherLayer.apply(z_t1), dim=0)
                z_t2 = torch.cat(GatherLayer.apply(z_t2), dim=0)
                z_t = torch.cat((z_t1, z_t2))

                # target projection head features
                z_idc_t1 = torch.cat(GatherLayer.apply(z_idc_t1), dim=0)
                z_idc_t2 = torch.cat(GatherLayer.apply(z_idc_t2), dim=0)
                z_idc_t = torch.cat((z_idc_t1, z_idc_t2))

                clip_z_t1 = torch.cat(GatherLayer.apply(clip_z_t1), dim=0)
                clip_z_t2 = torch.cat(GatherLayer.apply(clip_z_t2), dim=0)
                clip_z_t = torch.cat((clip_z_t1, clip_z_t2))

                y_target_1, y_target_2 = y_target.chunk(2)
                y_target_1 = torch.cat(GatherLayer.apply(y_target_1), dim=0)
                y_target_2 = torch.cat(GatherLayer.apply(y_target_2), dim=0)
                y_target = torch.cat((y_target_1, y_target_2))
            else:
                # target logits
                out_t = torch.cat(GatherLayer.apply(out_t), dim=0)
                # target contrastive features
                out_con_t = torch.cat(GatherLayer.apply(out_con_t), dim=0)
                # target projection head features
                z_t = torch.cat(GatherLayer.apply(z_t), dim=0)
                z_idc_t = torch.cat(GatherLayer.apply(z_idc_t), dim=0)
                clip_z_t = torch.cat(GatherLayer.apply(clip_z_t), dim=0)
                y_target = torch.cat(GatherLayer.apply(y_target), dim=0)

            y_source = torch.cat(GatherLayer.apply(y_source), dim=0)

            source_index = torch.cat(GatherLayer.apply(source_index), dim=0)
            target_index = torch.cat(GatherLayer.apply(target_index), dim=0)

            # consistency loss
            temp_t = out_t
            temp_s = out_s
            p = torch.mm(temp_t, temp_s.t())
            temp_t = F.normalize(out_con_t, dim=1)
            temp_s = F.normalize(out_con_s, dim=1)
            p_bottom = torch.mm(temp_t, temp_s.t()).float()
            threshold = self.args.consistency_threshold
            p_bottom[p_bottom > threshold] = 1
            p_bottom[p_bottom <= threshold] = 0
            consistency_loss = F.binary_cross_entropy_with_logits(p.view(-1), p_bottom.view(-1))

            if args.supervised_labels:
                y_target_est = y_target
                y = torch.cat((y_source, y_target), dim=0)
                selection = torch.ones(y.size(0), dtype=bool, device=self.device)
            else:
                # select all source data
                selection_source = torch.ones(y_source.size(0), dtype=bool, device=self.device)

                if args.oracle:
                    selection_target = torch.ones((out_t.size(0)), device=self.device, dtype=bool)
                    y_target_est = y_target
                else:
                    # select only target data that has a certain degree of confidence
                    selection_target = (
                        compute_entropy(F.softmax(out_t, dim=1))
                        < math.log(self.num_classes) / self.args.selection_factor
                    )
                    y_target_est = out_t.max(dim=1)[1]
                selection = torch.cat((selection_source, selection_target), dim=0)

                y = torch.cat((y_source, y_target_est), dim=0)

                # statistics about the target selection
                filter_stats = self.compute_filter_statistics(
                    y_target, y_target_est, selection_target
                )
                n_pseudo_labels = selection_target.sum().float().detach()

            feat = torch.cat((z_s, z_t), dim=0)

            # aug-based contrastive for target
            if args.target_2_augs:
                nce_loss_target_aug_based = self.aug_based_contrastive(z_t1, z_t2)

                b = z_t1.size(0)
                aux = torch.block_diag(*(2 * b * [torch.eye(args.n_clips)]))
                logit_mask = 1 - torch.block_diag(
                    *(2 * b * [torch.ones(args.n_clips, args.n_clips)])
                )
                logit_mask = torch.bitwise_or(logit_mask.int(), aux.int()).to(self.device)
                nce_loss_target_clip_aug_based = self.aug_based_contrastive(
                    clip_z_t1, clip_z_t2, logit_mask=logit_mask
                )

            else:
                nce_loss_target_aug_based = torch.tensor(0.0, device=self.device)
                nce_loss_target_clip_aug_based = torch.tensor(0.0, device=self.device)

            # label-based contrastive for  source and target
            nce_loss_source_label_based = self.label_based_contrastive(z_s, y_source)
            nce_loss_target_label_based = self.label_based_contrastive(
                z_t[selection_target], y_target_est[selection_target]
            )

            # inter domain contrastive
            if self.args.third_projection:
                nce_loss_inter_domain = self.inter_domain_based_contrastive(
                    z_idc_s, z_idc_t, y_source, y_target_est, selection
                )
            else:
                nce_loss_inter_domain = self.inter_domain_based_contrastive(
                    z_s, z_t, y_source, y_target_est, selection
                )

            # complete loss
            complete_nce = self.filtered_label_based_contrastive(feat, y, selection)

            # cross entropy weights
            ce_weight = self.args.ce_loss_weight
            ce_weight_target = self.args.ce_loss_target_weight
            # aug-based contrastive weight
            nce_loss_target_aug_based_weight = self.args.nce_loss_target_aug_based_weight
            nce_loss_target_clip_aug_based_weight = self.args.nce_loss_target_clip_aug_based_weight
            # label-based contrastive weights
            nce_loss_source_label_based_weight = self.args.nce_loss_source_label_based_weight
            nce_loss_target_label_based_weight = self.args.nce_loss_target_label_based_weight

            # inter domain contrastive weight
            nce_loss_inter_domain_weight = self.args.nce_loss_inter_domain_weight
            # consistency
            consistency_loss_weight = self.args.consistency_loss_weight
            # old complete loss
            complete_nce_weight = self.args.complete_nce_weight

            loss = (
                ce_weight * ce_loss
                + ce_weight_target * ce_loss_target
                + nce_loss_target_aug_based_weight * nce_loss_target_aug_based
                + nce_loss_target_clip_aug_based_weight * nce_loss_target_clip_aug_based
                + nce_loss_source_label_based_weight * nce_loss_source_label_based
                + nce_loss_target_label_based_weight * nce_loss_target_label_based
                + nce_loss_inter_domain_weight * nce_loss_inter_domain
                + complete_nce_weight * complete_nce
                + consistency_loss_weight * consistency_loss
            )

        acc1, acc5 = accuracy_at_k(out_s, y_source, top_k=(1, 5))
        log = {
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_ce_loss_source": ce_loss,
        }
        results = {
            "loss": loss,
            "out_s": out_s,
            "y_source": y_source,
        }

        if not args.source_only:
            # handle logging
            log.update(
                {
                    "train_ce_loss_target": ce_weight_target * ce_loss_target,
                    "train_nce_loss_target_aug_based": nce_loss_target_aug_based_weight
                    * nce_loss_target_aug_based,
                    "train_nce_loss_source_label_based": nce_loss_source_label_based_weight
                    * nce_loss_source_label_based,
                    "train_nce_loss_target_label_based": nce_loss_target_label_based_weight
                    * nce_loss_target_label_based,
                    "train_nce_loss_target_clip_aug_based": nce_loss_target_clip_aug_based_weight
                    * nce_loss_target_clip_aug_based,
                    "train_nce_loss_inter_domain": nce_loss_inter_domain_weight
                    * nce_loss_inter_domain,
                    "train_complete_nce": complete_nce_weight * complete_nce,
                    "train_consistency_loss": consistency_loss_weight * consistency_loss,
                    # pseudo label stuff
                    "train_acc_pseudo_label_passed": filter_stats["acc_pseudo_label_passed"],
                    "train_acc_pseudo_label_filtered": filter_stats["acc_pseudo_label_filtered"],
                    "train_acc_pseudo_label_total": filter_stats["acc_pseudo_label_total"],
                    "train_n_pseudo_labels": n_pseudo_labels,
                }
            )

            results.update({"out_t": out_t, "y_target": y_target, **filter_stats})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return results

    def validation_step(self, batch, batch_idx):
        X, target = batch

        output = self(X)
        loss = F.cross_entropy(output, target).detach()

        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        if dist.is_initialized():
            output = torch.cat(GatherLayer.apply(output), dim=0)
            target = torch.cat(GatherLayer.apply(target), dim=0)

        results = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "outputs": output,
            "targets": target,
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)

    def compute_filter_statistics(self, y_target, y_target_est, selection):
        acc_pseudo_label_passed = self.compute_acc(y_target[selection], y_target_est[selection])
        acc_pseudo_label_filtered = self.compute_acc(y_target[~selection], y_target_est[~selection])
        acc_pseudo_label_total = self.compute_acc(y_target, y_target_est)

        # entropies
        dist_per_pseudo_passed = self.compute_dist_per_class(
            y_target[selection], y_target_est[selection]
        )

        dist_per_pseudo_filtered = self.compute_dist_per_class(
            y_target[~selection], y_target_est[~selection]
        )
        dist_per_pseudo_total = self.compute_dist_per_class(y_target, y_target_est)

        hist_passed = self.histogram(y_target_est[selection])
        hist_filtered = self.histogram(y_target_est[~selection])
        hist_total = self.histogram(y_target_est)

        data = {
            "hist_passed": hist_passed,
            "hist_filtered": hist_filtered,
            "hist_total": hist_total,
            "acc_pseudo_label_passed": acc_pseudo_label_passed,
            "acc_pseudo_label_filtered": acc_pseudo_label_filtered,
            "acc_pseudo_label_total": acc_pseudo_label_total,
            "dist_per_pseudo_passed": dist_per_pseudo_passed,
            "dist_per_pseudo_filtered": dist_per_pseudo_filtered,
            "dist_per_pseudo_total": dist_per_pseudo_total,
        }
        return data

    def histogram(self, data):
        """Generates histogram from input data."""

        hist = torch.zeros(self.num_classes, device=self.device)

        for elem in data:
            hist[elem] += 1

        return hist

    def compute_acc(self, y, y_hat):
        """Computes accuracy."""

        acc = y == y_hat
        if len(acc) > 0:
            acc = acc.sum().detach().true_divide(acc.size(0))
        else:
            acc = torch.tensor(0.0, device=self.device)

        return acc

    def compute_dist_per_class(self, y, y_hat):
        """Computes distribution per class."""

        dist_per_pseudo = {}
        for c in range(self.num_classes):
            index = y_hat == c
            filtered_y = y[index]
            if len(filtered_y):
                ind, count = torch.unique(filtered_y, return_counts=True)
                count = count.float()

                dist = torch.zeros(self.num_classes, device=self.device)
                dist[ind] = count

                dist_per_pseudo[c] = dist
            else:
                dist_per_pseudo[c] = None

        return dist_per_pseudo


class PretrainVideoModel(VideoModel):
    def training_step(self, batch, batch_idx):
        X, y = batch

        output = self(X)
        loss = F.cross_entropy(output, y)

        acc1, acc5 = accuracy_at_k(output, y, top_k=(1, 5))

        results = {
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(results, on_epoch=True, sync_dist=True)
        return {"loss": loss, "out_s": output, "y_source": y}
