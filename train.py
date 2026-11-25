"""Train the full StageRelation method and select the highest validation QWK."""
import argparse
from pathlib import Path
import torch
import yaml
from stage_relation.data import GleamDataset, check_disjoint
from stage_relation.engine import (setup, make_model, make_loader, make_optimizer, train_epoch,
    evaluate, write_json, save_predictions, source_hashes, rng_state, restore_rng, save_checkpoint)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/train.yaml'))
    p.add_argument('--output', type=Path, default=Path('outputs/stage_relation'))
    p.add_argument('--data'); p.add_argument('--device', choices=['cpu', 'cuda'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.data: config['data'] = args.data
    if args.device: config['device'] = args.device
    for key in ('epochs', 'batch_size', 'image_size', 'feature_dim', 'grid_size', 'token_layers', 'token_heads', 'num_threads'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if config['num_workers'] < 0 or config['feature_dim'] % 8 or config['feature_dim'] % config['token_heads']:
        raise ValueError('Invalid workers or feature_dim/head divisibility')
    if not 0 <= config['noise_probability'] <= 1 or config['noise_std'] < 0:
        raise ValueError('Invalid noise policy')
    setup(config)
    train = GleamDataset(config['data'], 'trainset', config['image_size'], train=True)
    val = GleamDataset(config['data'], 'valset', config['image_size'])
    check_disjoint(train, val)
    if len({y for _, y in val.records}) < 2:
        raise ValueError('Validation QWK selection requires at least two ground-truth classes')
    if config['backbone'] == 'resnet50' and len(train) % config['batch_size'] == 1 and config['image_size'] <= 32:
        raise ValueError('Tiny ResNet inputs need batches >= 2 for BatchNorm')
    manifest = {'train': train.records, 'val': val.records}
    # JSON round trip normalizes tuples for checkpoint/config comparison.
    import json
    manifest = json.loads(json.dumps(manifest))
    hashes = source_hashes()
    state = None
    if args.resume:
        state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        if state['config'] != config or state['sources'] != hashes or state['manifest'] != manifest:
            raise ValueError('Resume requires the same configuration, sources and eye labels')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    model = make_model(config, initialize=not args.resume)
    optimizer = make_optimizer(model, config)
    train_loader = make_loader(train, config, train=True); val_loader = make_loader(val, config)
    history, best, start = [], -float('inf'), 1
    if state:
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        history, best, start = state['history'], state['best_qwk'], state['epoch'] + 1
        restore_rng(state['rng']); del state
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'source_hashes.json', hashes)
    write_json(args.output / 'data_manifest.json', manifest)
    for epoch in range(start, config['epochs'] + 1):
        training = train_epoch(model, train_loader, optimizer, config, epoch)
        validation, _ = evaluate(model, val_loader, config['device'])
        score = validation['qwk']
        if score is None: raise ValueError('Validation QWK is undefined')
        row = {'epoch': epoch, 'train': training, 'validation': validation}
        history.append(row)
        if score > best:
            best = score
            save_checkpoint(args.output / 'best.pt', {'model': model.state_dict(), 'config': config,
                'epoch': epoch, 'validation': validation, 'sources': hashes})
        save_checkpoint(args.output / 'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'config': config, 'epoch': epoch, 'history': history, 'best_qwk': best,
            'rng': rng_state(), 'sources': hashes, 'manifest': manifest})
        write_json(args.output / 'history.json', history)
        print(f"epoch={epoch}/{config['epochs']} loss={training['total']:.4f} val_qwk={score:.4f}", flush=True)
    checkpoint = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics, predictions = evaluate(model, val_loader, config['device'])
    write_json(args.output / 'validation.json', {'selected_epoch': checkpoint['epoch'], **metrics})
    save_predictions(args.output / 'validation_predictions.csv', predictions)
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
