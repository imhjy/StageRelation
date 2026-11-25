"""Evaluate a saved model on a clean labeled split without downloading weights."""
import argparse
from pathlib import Path
import torch
import yaml
from stage_relation.data import GleamDataset
from stage_relation.engine import setup, make_model, make_loader, evaluate, write_json, save_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, help='Model YAML; required for a raw state_dict')
    p.add_argument('--check-only', action='store_true', help='Load and run synthetic inputs without a dataset')
    p.add_argument('--data'); p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--split', choices=['valset', 'testset'], default='valset')
    p.add_argument('--output', type=Path, default=Path('outputs/validation'))
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint and 'config' in checkpoint:
        config = dict(checkpoint['config'])
        state = checkpoint['model']
        checkpoint_epoch = checkpoint.get('epoch')
        checkpoint_format = 'checkpoint'
    elif isinstance(checkpoint, dict) and checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        if args.config is None:
            p.error('Raw state_dict requires --config configs/train.yaml')
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        state = checkpoint
        checkpoint_epoch = None
        checkpoint_format = 'raw_state_dict'
    else:
        raise ValueError('Expected a raw state_dict or a checkpoint containing model and config')
    config['device'] = args.device
    if args.data: config['data'] = args.data
    setup(config)
    model = make_model(config, initialize=False)
    status = model.load_state_dict(state, strict=True)
    if args.check_only:
        model.eval()
        generator = torch.Generator().manual_seed(20261125)
        images = tuple(torch.randn(1, 3, config['image_size'], config['image_size'], generator=generator)
                       .to(config['device']) for _ in range(3))
        with torch.inference_mode():
            logits = model(*images)
        if logits.shape != (1, 4) or not torch.isfinite(logits).all():
            raise ValueError('Invalid output shape or non-finite model output')
        result = {'checkpoint': str(args.checkpoint.resolve()), 'format': checkpoint_format,
                  'strict_load': True, 'missing_keys': status.missing_keys,
                  'unexpected_keys': status.unexpected_keys, 'output_shape': list(logits.shape),
                  'finite_output': True, 'input_type': 'synthetic',
                  'probabilities': logits.softmax(-1)[0].cpu().tolist()}
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / 'load_check.json', result)
        print(result)
        return
    dataset = GleamDataset(config['data'], args.split, config['image_size'])
    metrics, predictions = evaluate(model, make_loader(dataset, config), config['device'])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'metrics.json', {'checkpoint_epoch': checkpoint_epoch, 'checkpoint_format': checkpoint_format, 'split': args.split, **metrics})
    save_predictions(args.output / 'predictions.csv', predictions)
    print(metrics)


if __name__ == '__main__':
    main()
