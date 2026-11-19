import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    edge_mask = torch.abs(mask - F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1))
    edge_pred = torch.abs(pred - F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1))
    edge_bce = F.binary_cross_entropy_with_logits(edge_pred, edge_mask, reduce='none')

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou + edge_bce).mean()
