import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from data import DEFAULT_DATA, resize_with_padding


INK = '#172842'
GRAY = '#4f6179'
BLUE = '#2d6292'
GREEN = '#08796f'
ORANGE = '#ba682d'
LINE = '#cbd5e1'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': GRAY, 'ytick.color': GRAY, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': LINE})


def save(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=175, facecolor='white', bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)


def architecture(output, settings):
    fig, ax = plt.subplots(figsize=(12, 7.3))
    ax.set(xlim=(0, 12), ylim=(0, 7.25))
    ax.axis('off')
    ax.text(0.45, 6.88, 'U-Net for pedestrian segmentation', fontsize=23, weight='bold')
    ax.text(0.45, 6.48, 'Feature maps are shown as channels × height × width.', fontsize=11, color=GRAY)

    def box(x, y, label, shape, color, fill, width=2.8):
        block = FancyBboxPatch(
            (x - width / 2, y - 0.29), width, 0.58, boxstyle='round,pad=0.023,rounding_size=0.095',
            facecolor=fill, edgecolor=color, linewidth=1.15,
        )
        ax.add_patch(block)
        ax.text(x, y + 0.077, label, color=color, fontsize=12, weight='bold', ha='center', va='center')
        ax.text(x, y - 0.124, shape, color=INK, fontsize=10, ha='center', va='center')

    def arrow(start, stop, color=GRAY, dashed=False):
        ax.add_patch(FancyArrowPatch(start, stop, arrowstyle='-|>', mutation_scale=14, lw=1.2,
                                    linestyle='--' if dashed else '-', color=color, shrinkA=1, shrinkB=1))

    left, right = 2.55, 9.45
    levels = (4.99, 3.85, 2.71)
    box(left, 5.96, 'RGB input', '3 × 128 × 128', BLUE, '#eef4fb')
    box(right, 5.96, '1 × 1 head + sigmoid', '1 × 128 × 128 probability map', GREEN, '#e8f5f1', 3.16)
    for level, y in enumerate(levels, 1):
        channels = 16 * 2 ** (level - 1)
        size = 128 // 2 ** (level - 1)
        shape = f'{channels} × {size} × {size}'
        box(left, y, f'Encoder {level}', shape, BLUE, '#eef4fb')
        box(right, y, f'Decoder {level}', shape, GREEN, '#e8f5f1')
        arrow((left + 1.42, y), (right - 1.43, y), BLUE, dashed=True)
        ax.text(6, y + 0.13, 'Skip concatenation', ha='center', color=BLUE, fontsize=9.7)

    box(6, 1.55, 'Bottleneck', '128 × 16 × 16', ORANGE, '#fff3e8', 3.18)
    arrow((left, 5.65), (left, 5.3))
    arrow((right, 5.3), (right, 5.65))
    for upper, lower in zip(levels, levels[1:]):
        arrow((left, upper - 0.31), (left, lower + 0.31))
        arrow((right, lower + 0.31), (right, upper - 0.31))
        ax.text(left + 0.14, (upper + lower) / 2, '2 × 2 max pool', fontsize=8.9, color=GRAY, va='center')
        ax.text(right + 0.14, (upper + lower) / 2, 'Bilinear ×2', fontsize=8.9, color=GRAY, va='center')
    arrow((left, levels[-1] - 0.31), (4.39, 1.66))
    arrow((7.61, 1.66), (right, levels[-1] - 0.31))
    ax.text(2.26, 1.77, '2 × 2 max pool', color=GRAY, fontsize=9)
    ax.text(8.63, 1.87, 'Bilinear ×2', color=GRAY, fontsize=9)

    ax.plot([0.45, 11.55], [0.86, 0.86], color=LINE, lw=1)
    ax.text(0.45, 0.56, 'Encoder, decoder and bottleneck blocks use two 3 × 3 convolutions,\n'
            'each followed by GroupNorm (4 groups) and ReLU.', fontsize=10.1, va='center', linespacing=1.35)
    ax.text(11.55, 0.53, f"Threshold {settings['threshold']:.2f}  ·  {settings['parameters']:,} trainable parameters",
            fontsize=10, color=GRAY, ha='right', va='center')
    save(fig, output, 'architecture')


def training_curves(output, rows, settings):
    epochs = [int(row['epoch']) for row in rows]
    loss = [float(row['train_loss']) for row in rows]
    iou = [float(row['validation_iou_at_0.5']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), layout='constrained')
    fig.suptitle('Training and validation', fontsize=18, fontweight='bold')
    ticks = np.unique(np.linspace(1, max(epochs), 6, dtype=int))
    for ax, y, label, color in zip(axes, (loss, iou), ('Training loss', 'Validation foreground IoU at 0.5'), (BLUE, GREEN)):
        ax.plot(epochs, y, color=color, lw=2.2, marker='o', ms=3.2)
        ax.set(xlabel='Epoch', ylabel=label, xticks=ticks)
        ax.grid(axis='y', color='#e4eaf0', alpha=0.9)
        ax.set_axisbelow(True)
    best = settings['best_epoch']
    score = settings['best_validation_iou_at_0.5']
    axes[1].set_ylim(0, max(iou) + 0.12)
    axes[1].annotate(f'Best {score:.3f} at epoch {best}', xy=(best, score), xytext=(-7, 17),
                     textcoords='offset points', ha='right', color=GREEN, fontsize=10,
                     arrowprops={'arrowstyle': '->', 'color': GREEN})
    save(fig, output, 'training_curves')


def mask_examples(output, root, run, size):
    with (run / 'test_by_image.csv').open(newline='', encoding='utf-8') as file:
        ordered = sorted(csv.DictReader(file), key=lambda row: float(row['iou']))
    examples = [ordered[-1], ordered[len(ordered) // 2], ordered[0]]
    with np.load(run / 'test_masks.npz') as cached:
        names = list(cached['names'])
        prediction, target, valid = (cached[key] for key in ('prediction', 'target', 'valid'))

    fig, axes = plt.subplots(3, 4, figsize=(10.6, 9.3))
    fig.subplots_adjust(left=0.035, right=0.98, top=0.89, bottom=0.11, hspace=0.24, wspace=0.07)
    fig.suptitle('Held-out pedestrian masks', fontsize=18, fontweight='bold')
    for ax, label in zip(axes[0], ('Input', 'Ground truth', 'U-Net prediction', 'False pixels')):
        ax.set_title(label, fontsize=11, weight='bold', pad=8)
    for row, result, label in zip(axes, examples, ('Highest IoU', 'Median IoU', 'Lowest IoU')):
        name = result['image']
        idx = names.index(name)
        with Image.open(root / 'PNGImages' / f'{name}.png') as image:
            rgb, _, _, (left, top, right, bottom) = resize_with_padding(image, size=size)
        a = rgb[top:bottom, left:right]
        expected = target[idx, top:bottom, left:right]
        estimated = prediction[idx, top:bottom, left:right]
        inside = valid[idx, top:bottom, left:right]
        errors = a.astype(float) / 255 * 0.54 + 0.20
        errors[estimated & ~expected & inside] = (0.93, 0.37, 0.18)
        errors[expected & ~estimated & inside] = (0.18, 0.68, 0.86)
        row[0].imshow(a)
        row[1].imshow(expected, cmap=ListedColormap(['#eef2f6', '#123e64']), vmin=0, vmax=1)
        row[2].imshow(estimated, cmap=ListedColormap(['#eef2f6', '#08796f']), vmin=0, vmax=1)
        row[3].imshow(errors.clip(0, 1))
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
        row[0].text(0.5, -0.055, f"{label} · {name} · {float(result['iou']):.3f}", fontsize=8.2,
                    transform=row[0].transAxes, ha='center', va='top')
    fig.supxlabel('Orange = extra predicted pixels     Blue = missed pedestrian pixels', fontsize=10, y=0.025)
    save(fig, output, 'held_out_masks')


def main():
    parser = argparse.ArgumentParser(description='Draw the architecture, learning curves and optional test masks.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--run', type=Path, default=Path('runs/penn-fudan'))
    parser.add_argument('--output', type=Path, default=Path('runs/penn-fudan/figures'))
    parser.add_argument('--examples', action='store_true', help='Include examples with original dataset pixels.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((args.run / 'metrics.json').read_text(encoding='utf-8'))
    with (args.run / 'history.csv').open(newline='', encoding='utf-8') as file:
        history = list(csv.DictReader(file))
    settings = experiment['settings']
    architecture(args.output, settings)
    training_curves(args.output, history, settings)
    if args.examples:
        mask_examples(args.output, args.data, args.run, settings['image_size'])
    print(f'Saved figures to {args.output}.')


if __name__ == '__main__':
    main()
