import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import image_tensor, resize_with_padding
from model import UNet


def main():
    parser = argparse.ArgumentParser(description='Predict the pedestrian mask of one RGB image.')
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('results/unet.pt'))
    parser.add_argument('--metrics', type=Path, default=Path('results/metrics.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/prediction.png'))
    args = parser.parse_args()

    torch.set_num_threads(4)
    settings = json.loads(args.metrics.read_text(encoding='utf-8'))['settings']
    model = UNet().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    with Image.open(args.image) as image:
        rgb, _, _, box = resize_with_padding(image, size=settings['image_size'])
        original_size = image.size
    with torch.inference_mode():
        confidence = model(image_tensor(rgb).unsqueeze(0)).sigmoid().numpy()[0, 0]
    mask = Image.fromarray(((confidence >= settings['threshold']) * 255).astype(np.uint8))
    mask = mask.crop(box).resize(original_size, Image.Resampling.NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(args.output)
    print(f'Saved {mask.width} × {mask.height} pedestrian mask to {args.output}')


if __name__ == '__main__':
    main()
