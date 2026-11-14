import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DEFAULT_DATA, IMAGE_SIZE, PennFudan, dataset_split
from metrics import best_threshold, counts, predict_loader, scores
from model import UNet, segmentation_loss


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, target, valid in loader:
        images = images.to(device, memory_format=torch.channels_last)
        target, valid = target.to(device), valid.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(images), target, valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


def positional_prior(dataset):
    target = torch.stack(dataset.targets).sum(dim=0)
    valid = torch.stack(dataset.valid).sum(dim=0)
    return (target / valid.clamp(min=1)).numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='Train U-Net and evaluate the held-out Penn–Fudan images.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=Path('runs/penn-fudan'))
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error('Epochs, batch size and threads must be at least 1.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Choose --device cpu.')

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    split = dataset_split(args.data, args.seed)
    write_json(args.out / 'split.json', split)

    train = PennFudan(args.data, split['train'], augment=True)
    validation = PennFudan(args.data, split['validation'])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation, batch_size=args.batch_size)

    model = UNet().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0001)
    checkpoint = args.out / 'unet.pt'
    best_iou = -1.0
    best_epoch = 0

    print('Split:', ', '.join(f'{key}={len(value)}' for key, value in split.items()), flush=True)
    with (args.out / 'history.csv').open('w', newline='', encoding='utf-8') as file:
        history = csv.writer(file)
        history.writerow(['epoch', 'train_loss', 'validation_iou_at_0.5', 'learning_rate'])
        for epoch in range(1, args.epochs + 1):
            learning_rate = optimizer.param_groups[0]['lr']
            loss = train_epoch(model, train_loader, optimizer, device)
            probability, target, valid = predict_loader(model, validation_loader, device)
            validation_score = scores(counts(probability >= 0.5, target, valid).sum(axis=0))['iou']
            history.writerow([epoch, loss, validation_score, learning_rate])
            file.flush()
            if validation_score > best_iou:
                best_iou, best_epoch = validation_score, epoch
                torch.save(model.state_dict(), checkpoint)
            scheduler.step()
            print(f'Epoch {epoch:02}/{args.epochs}  loss={loss:.4f}  validation IoU={validation_score:.4f}', flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    probability, target, valid = predict_loader(model, validation_loader, device)
    threshold, validation_curve = best_threshold(probability, target, valid)
    prior = positional_prior(train)
    prior_threshold, prior_curve = best_threshold(prior, target, valid)

    test = PennFudan(args.data, split['test'])
    probability, target, valid = predict_loader(model, DataLoader(test, batch_size=args.batch_size), device)
    prediction = probability >= threshold
    methods = {
        'unet': counts(prediction, target, valid),
        'positional_prior': counts(prior >= prior_threshold, target, valid),
        'all_background': counts(np.zeros_like(target), target, valid),
    }
    groups = {campus: [name.startswith(campus + 'Ped') for name in test.names] for campus in ('Penn', 'Fudan')}
    experiment = {
        'dataset': 'Penn–Fudan Pedestrian Database',
        'split': {key: len(value) for key, value in split.items()},
        'test_groups': {campus: sum(selected) for campus, selected in groups.items()},
        'python': sys.version.split()[0],
        'torch': str(torch.__version__),
        'settings': {
            'seed': args.seed, 'image_size': IMAGE_SIZE, 'epochs': args.epochs,
            'batch_size': args.batch_size, 'threads': args.threads, 'device': args.device,
            'optimizer': 'AdamW', 'learning_rate': 0.001, 'min_learning_rate': 0.0001,
            'weight_decay': 0.0001, 'loss': '0.5 × BCE + 0.5 × Dice loss',
            'channels': [16, 32, 64, 128], 'group_norm_groups': 4,
            'parameters': sum(p.numel() for p in model.parameters()),
            'best_epoch': best_epoch, 'best_validation_iou_at_0.5': best_iou,
            'threshold': threshold, 'positional_prior_threshold': prior_threshold,
        },
        'validation_threshold_scores': {'unet': validation_curve, 'positional_prior': prior_curve},
        'test': {name: scores(pixel_counts.sum(axis=0)) for name, pixel_counts in methods.items()},
        'test_by_campus': {
            campus: scores(methods['unet'][selected].sum(axis=0)) for campus, selected in groups.items()
        },
    }
    write_json(args.out / 'metrics.json', experiment)
    blocks = [(name, sum(p.numel() for p in layer.parameters())) for name, layer in model.named_children()]
    with (args.out / 'model_parameters.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['block', 'trainable_parameters'])
        writer.writerows(blocks)
        writer.writerow(['total', experiment['settings']['parameters']])
    with (args.out / 'test_by_image.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['image', *scores((0, 0, 0, 0)).keys()])
        writer.writeheader()
        for name, pixel_counts in zip(test.names, methods['unet']):
            writer.writerow({'image': name, **scores(pixel_counts)})
    np.savez_compressed(args.out / 'test_masks.npz', names=test.names, prediction=prediction, target=target, valid=valid)
    for name, outcome in experiment['test'].items():
        print(f'{name:17s}  IoU={outcome["iou"]:.4f}  Dice={outcome["dice"]:.4f}  accuracy={outcome["accuracy"]:.4f}')


if __name__ == '__main__':
    main()
