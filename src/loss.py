import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(logits, pos_mask, logit_mask=None, temperature=0.2):
    """Computes the NCE loss.

    Args:
        logits: similarity matrix of size n by n
        pos_mask: mask of 1s when the elements are positives
        logit_mask (optional): If not provided, consider all instances as for the denominator.
            Otherwise, consider only selected elements. This is done line-wise. Defaults to None.
        temperature (float, optional): [description]. Defaults to 0.2.

    Returns:
        NCE loss.
    """
    b = logits.size(0)
    device = logits.device

    logits = logits / temperature

    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # all matches excluding the main diagonal
    if logit_mask is None:
        logit_mask = ~torch.eye(b, dtype=bool, device=device)

    div = torch.sum(torch.exp(logits) * logit_mask, dim=1, keepdim=True)
    log_prob = torch.log(torch.exp(logits) / div)

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)

    # filter where there are no positives
    indexes = pos_mask.sum(1) > 0
    pos_mask = pos_mask[indexes]
    mean_log_prob_pos = mean_log_prob_pos[indexes]
    mean_log_prob_pos /= pos_mask.sum(1)

    if len(mean_log_prob_pos):
        loss = -mean_log_prob_pos.mean()
    else:
        loss = torch.tensor(0.0, device=device)

    return loss


# nce based on augmentations
class AugBasedNCE(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, x1, x2, logit_mask=None):
        device = x1.device
        n_imgs = x1.size(0)
        n_augs = 2

        x = torch.cat((x1, x2), dim=0)
        x = F.normalize(x, dim=1)
        logits = torch.mm(x, x.t())

        helper = torch.eye(n_imgs, dtype=bool, device=device)
        temp = torch.cat([helper for i in range(n_augs)])
        pos_mask = torch.cat([temp for i in range(n_augs)], dim=1)
        pos_mask.fill_diagonal_(False)
        return info_nce(logits, pos_mask, logit_mask=logit_mask, temperature=self.temperature)


# nce based on labels
class LabelBasedNCE(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, x, y):
        if not len(x):
            device = x.device
            return torch.tensor(0.0, device=device)

        b = x.size(0)

        x = F.normalize(x, dim=1)
        logits = torch.mm(x, x.t())

        labels_matrix = y.reshape(1, -1).repeat(b, 1)
        pos_mask = labels_matrix == labels_matrix.t()
        pos_mask.fill_diagonal_(False)
        return info_nce(logits, pos_mask, temperature=self.temperature)


# nce based on filtered labels
class FilteredLabelBasedNCE(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, x, y, selection):
        x = x[selection]
        y = y[selection]

        b = x.size(0)

        x = F.normalize(x, dim=1)
        logits = torch.mm(x, x.t())

        labels_matrix = y.reshape(1, -1).repeat(b, 1)
        pos_mask = labels_matrix == labels_matrix.t()
        pos_mask.fill_diagonal_(False)
        return info_nce(logits, pos_mask, temperature=self.temperature)


# nce inter-domain
class InterDomainBasedNCE(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, xs, xt, ys, yt, selection=None):
        bs = xs.size(0)
        device = xs.device

        x = torch.cat((xs, xt), dim=0)
        y = torch.cat((ys, yt), dim=0)
        if selection is not None:
            x = x[selection]
            y = y[selection]

        b = x.size(0)

        x = F.normalize(x, dim=1)
        logits = torch.mm(x, x.t())

        labels_matrix = y.reshape(1, -1).repeat(b, 1)
        pos_mask = labels_matrix == labels_matrix.t()
        pos_mask.fill_diagonal_(False)
        pos_mask[:bs, :bs] = False
        pos_mask[bs:, bs:] = False

        logit_mask = torch.ones_like(logits, dtype=bool, device=xs.device)
        logit_mask.fill_diagonal_(False)
        logit_mask[:bs, :bs] = False
        logit_mask[bs:, bs:] = False

        if (logit_mask == 0).all():
            loss = torch.tensor(0.0, device=device)
        else:
            loss = info_nce(logits, pos_mask, logit_mask=logit_mask, temperature=self.temperature)
        return loss
