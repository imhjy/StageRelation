"""Strict GLEAM reader; one eye per directory, three complete modalities."""
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

FILES = ('SLO.jpg', 'Thickness.jpg', 'VF.jpg')


class GleamDataset(Dataset):
    def __init__(self, root, split, image_size=224, train=False):
        self.root, self.split, self.image_size, self.train = Path(root), split, image_size, train
        self.jitter = T.ColorJitter(brightness=.1, contrast=.1)
        self.normalize = T.Normalize((.485, .456, .406), (.229, .224, .225))
        with (self.root / (split + 'label.csv')).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'ID', 'Label'} <= set(reader.fieldnames or []):
                raise ValueError('CSV must contain ID,Label columns')
            self.records = [(str(int(r['ID'])), int(r['Label'])) for r in reader]
        self.records.sort(key=lambda r: int(r[0]))
        ids = [r[0] for r in self.records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'{split}: empty split or duplicate eye IDs')
        for eye, label in self.records:
            if label not in range(4):
                raise ValueError(f'{eye}: label must be 0..3')
            for name in FILES:
                path = self.root / split / eye / name
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        eye, label = self.records[index]
        images = []
        for name in FILES:
            with Image.open(self.root / self.split / eye / name) as im:
                images.append(im.convert('RGB').resize((self.image_size, self.image_size)))
        if self.train and torch.rand(()) < .5:
            images = [TF.hflip(im) for im in images]
        if self.train:
            images = [self.jitter(im) for im in images]
        return (*[self.normalize(TF.to_tensor(im)) for im in images], label, int(eye))


def check_disjoint(*datasets):
    seen = set()
    for dataset in datasets:
        ids = {eye for eye, _ in dataset.records}
        if seen & ids:
            raise ValueError('Overlapping eye IDs across splits: ' + str(sorted(seen & ids)))
        seen.update(ids)
