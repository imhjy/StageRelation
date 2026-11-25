"""Training and validation utilities; validation never applies quality corruption."""
import csv
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .model import StageRelation
from .losses import quality_augmentation, training_loss
from .metrics import classification_metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def setup(config):
    torch.set_num_threads(config['num_threads'])
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if config['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu or install a CUDA build of PyTorch.')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(config, initialize=True):
    model = StageRelation(**{k: config[k] for k in
        ('feature_dim', 'backbone', 'grid_size', 'token_layers', 'token_heads')},
        pretrained=initialize and config['pretrained_encoder'])
    if config['freeze_encoder']:
        model.encoder.freeze_backbones()
    return model.to(config['device'])


def make_loader(dataset, config, train=False):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=train,
        num_workers=config['num_workers'], pin_memory=config['device'].startswith('cuda'))


def make_optimizer(model, config):
    backbone, new = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (backbone if '.backbone.' in name else new).append(p)
    groups = []
    for params, key in [(backbone, 'learning_rate'), (new, 'new_module_learning_rate')]:
        if params:
            groups.append({'params': params, 'lr': config[key], 'base_lr': config[key]})
    return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])


def train_epoch(model, loader, optimizer, config, epoch):
    model.train(); sums = {}; count = 0
    for step, batch in enumerate(loader):
        images = tuple(x.to(config['device']) for x in batch[:3])
        labels = batch[3].to(config['device'])
        images, quality = quality_augmentation(images, config['noise_probability'], config['noise_std'])
        factor = .1 + .9 * (1 + math.cos(math.pi *
            ((epoch - 1) + (step + 1) / len(loader)) / config['epochs'])) / 2
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(*images, return_aux=True)
        loss, components = training_loss(logits, aux, labels, quality)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss: epoch {epoch}, batch {step}')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        count += len(labels)
        for key, value in {'total': loss, **components}.items():
            sums[key] = sums.get(key, 0.) + float(value.detach()) * len(labels)
    return {key: value / count for key, value in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); labels = []; probabilities = []; ids = []; loss_sum = 0.
    for batch in loader:
        y = batch[3].to(device)
        logits = model(*(x.to(device) for x in batch[:3]))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite validation output')
        loss_sum += float(F.cross_entropy(logits, y, reduction='sum'))
        labels.extend(y.cpu().tolist()); probabilities.extend(logits.softmax(-1).cpu().tolist())
        ids.extend(batch[4].tolist())
    result = classification_metrics(labels, probabilities)
    result['ce_loss'] = loss_sum / len(labels)
    return result, (ids, labels, probabilities)


def save_predictions(path, values):
    ids, labels, probabilities = values
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['ID', 'Label', 'Prediction', 'p0', 'p1', 'p2', 'p3'])
        writer.writerows((eye, y, int(np.argmax(p)), *p) for eye, y, p in zip(ids, labels, probabilities))


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'train.py', root / 'validate.py', *sorted((root / 'stage_relation').glob('*.py'))]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, payload):
    path = Path(path); temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); temporary.replace(path)
