import numpy as np
import torch


THRESHOLDS = tuple(n / 20 for n in range(4, 17))


@torch.inference_mode()
def predict_loader(model, loader, device):
    maps, masks, valid_pixels = [], [], []
    model.eval()
    for images, target, valid in loader:
        images = images.to(device, memory_format=torch.channels_last)
        maps.append(model(images).sigmoid().cpu().numpy()[:, 0])
        masks.append(target.numpy()[:, 0].astype(bool))
        valid_pixels.append(valid.numpy()[:, 0].astype(bool))
    return tuple(np.concatenate(items) for items in (maps, masks, valid_pixels))


def counts(prediction, target, valid):
    tp = (prediction & target & valid).sum(axis=(1, 2))
    fp = (prediction & ~target & valid).sum(axis=(1, 2))
    fn = (~prediction & target & valid).sum(axis=(1, 2))
    tn = (~prediction & ~target & valid).sum(axis=(1, 2))
    return np.stack((tp, fp, fn, tn), axis=1)


def scores(pixel_counts):
    tp, fp, fn, tn = (int(n) for n in pixel_counts)
    return {
        'iou': tp / max(tp + fp + fn, 1),
        'dice': 2 * tp / max(2 * tp + fp + fn, 1),
        'precision': tp / max(tp + fp, 1),
        'recall': tp / max(tp + fn, 1),
        'accuracy': (tp + tn) / max(tp + fp + fn + tn, 1),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def best_threshold(probability, target, valid):
    validation_curve = {}
    for threshold in THRESHOLDS:
        totals = counts(probability >= threshold, target, valid).sum(axis=0)
        validation_curve[threshold] = scores(totals)['iou']
    threshold = max(THRESHOLDS, key=lambda t: (validation_curve[t], -abs(t - 0.5), t))
    return threshold, validation_curve
