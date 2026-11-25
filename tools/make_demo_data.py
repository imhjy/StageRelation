"""Generate deterministic synthetic JPGs in the real GLEAM directory format."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/demo'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    for split, start, count in [('trainset', 1, 16), ('valset', 101, 8)]:
        with (args.output / (split + 'label.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['ID', 'Label'])
            for i in range(count):
                eye, label = start + i, i % 4
                folder = args.output / split / str(eye); folder.mkdir(parents=True)
                writer.writerow([eye, label])
                for name in ('SLO.jpg', 'Thickness.jpg', 'VF.jpg'):
                    array = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
                    Image.fromarray(array).save(folder / name)
    print('Synthetic data:', args.output.resolve())


if __name__ == '__main__':
    main()
