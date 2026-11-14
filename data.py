import argparse
import hashlib
import random
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DATASET_URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
DATASET_SHA256 = '9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2'
DEFAULT_DATA = Path('data/PennFudanPed')
IMAGE_SIZE = 128


def dataset_split(root, seed):
    groups = {'Fudan': [], 'Penn': []}
    for file in sorted((root / 'PNGImages').glob('*.png')):
        campus = next((name for name in groups if file.stem.startswith(name + 'Ped')), None)
        if campus is None or not (root / 'PedMasks' / f'{file.stem}_mask.png').is_file():
            raise ValueError(f'Unexpected image or missing mask: {file.name}')
        groups[campus].append(file.stem)
    if [len(group) for group in groups.values()] != [74, 96]:
        raise ValueError(f'Expected 170 image-mask pairs in {root}. Run python data.py first.')

    split = {'train': [], 'validation': [], 'test': []}
    generator = random.Random(seed)
    for names in groups.values():
        generator.shuffle(names)
        held_out = round(len(names) * 0.15)
        split['validation'].extend(names[:held_out])
        split['test'].extend(names[held_out:2 * held_out])
        split['train'].extend(names[2 * held_out:])
    return {part: sorted(names) for part, names in split.items()}


def resize_with_padding(image, mask=None, size=IMAGE_SIZE):
    width, height = image.size
    scale = size / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    left = (size - new_size[0]) // 2
    top = (size - new_size[1]) // 2
    box = (left, top, left + new_size[0], top + new_size[1])
    columns = slice(box[0], box[2])
    rows = slice(box[1], box[3])

    rgb = np.full((size, size, 3), 128, dtype=np.uint8)
    rgb[rows, columns] = image.convert('RGB').resize(new_size, Image.Resampling.BILINEAR)
    valid = np.zeros((size, size), dtype=np.uint8)
    valid[rows, columns] = 1

    target = np.zeros_like(valid)
    if mask is not None:
        if mask.size != image.size:
            raise ValueError('Image and mask dimensions differ.')
        labels = mask.resize(new_size, Image.Resampling.NEAREST)
        target[rows, columns] = np.asarray(labels) > 0
    return rgb, target, valid, box


def image_tensor(pixels):
    return torch.tensor(pixels, dtype=torch.float32).permute(2, 0, 1).div_(127.5).sub_(1)


class PennFudan(Dataset):
    def __init__(self, root, names, augment=False):
        self.names = names
        self.augment = augment
        self.images = []
        self.targets = []
        self.valid = []
        for name in names:
            with Image.open(root / 'PNGImages' / f'{name}.png') as image, \
                 Image.open(root / 'PedMasks' / f'{name}_mask.png') as mask:
                rgb, target, valid, _ = resize_with_padding(image, mask)
            self.images.append(image_tensor(rgb))
            self.targets.append(torch.from_numpy(target).unsqueeze(0).float())
            self.valid.append(torch.from_numpy(valid).unsqueeze(0).float())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image, target, valid = self.images[index], self.targets[index], self.valid[index]
        if self.augment:
            if random.random() < 0.5:
                image, target, valid = (tensor.flip(-1) for tensor in (image, target, valid))
            contrast = random.uniform(0.85, 1.15)
            brightness = random.uniform(-0.16, 0.16)
            image = ((image * contrast + brightness).clamp(-1, 1) * valid) + image * (1 - valid)
        return image, target, valid


def download(destination):
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'PennFudanPed'
    if target.exists():
        dataset_split(target, 42)
        print(f'Dataset already available at {target}.')
        return

    with tempfile.TemporaryDirectory(dir=destination) as folder:
        archive = Path(folder) / 'PennFudanPed.zip'
        digest = hashlib.sha256()
        with urlopen(DATASET_URL, timeout=30) as response, archive.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
        if digest.hexdigest() != DATASET_SHA256:
            raise ValueError('Dataset checksum differs from the published experiment archive.')
        with ZipFile(archive) as package:
            package.extractall(folder)
        extracted = Path(folder) / target.name
        dataset_split(extracted, 42)
        shutil.move(str(extracted), target)
    print(f'Downloaded 170 image-mask pairs to {target}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the Penn–Fudan dataset.')
    parser.add_argument('--output', type=Path, default=DEFAULT_DATA.parent)
    download(parser.parse_args().output)
