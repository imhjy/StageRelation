"""Compare extracted model and loss against an unpacked original source tree."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_relation.model import StageRelation
from stage_relation.losses import training_loss


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                               and n.name in names], type_ignores=[])
    exec(compile(selected, str(path), 'exec'), namespace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('reference_check.json'))
    args = p.parse_args()
    torch.set_num_threads(2)
    scope = dict(torch=torch, nn=nn, F=F, copy=copy, resnet50=resnet50, ResNet50_Weights=ResNet50_Weights)
    load_definitions(args.source / 'Model/gleam_proposals/common.py',
        {'mlp', 'transformer', 'present_mask', 'ImageEncoder', 'MultiModalEncoder', 'ProposalModel'}, scope)
    load_definitions(args.source / 'Model/gleam_proposals/a_methods.py', {'A02StageAdaptiveRelation'}, scope)
    load_definitions(args.source / 'Model/gleam_proposals/losses.py', {'method_loss'}, scope)
    checks = []
    for backbone in ('tiny', 'resnet50'):
        kwargs = dict(feature_dim=32, backbone=backbone, pretrained=False, grid_size=2, token_layers=1, token_heads=4)
        original = scope['A02StageAdaptiveRelation'](**kwargs)
        minimal = StageRelation(**kwargs)
        minimal.load_state_dict(original.state_dict(), strict=True)
        x = tuple(torch.randn(2, 3, 32, 32) for _ in range(3))
        labels = torch.tensor([0, 3]); quality = torch.tensor([[1., .1, 1.], [.1, 1., 1.]])
        original.eval(); minimal.eval()
        with torch.no_grad():
            old, old_aux = original(*x, return_aux=True)
            new, aux = minimal(*x, return_aux=True)
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            expected = scope['method_loss']('A02_StageRelation', old, old_aux, labels, {'quality': quality})
            actual, _ = training_loss(new, aux, labels, quality)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if backbone == 'tiny':
            original.train(); minimal.train()
            torch.manual_seed(91)
            old, old_aux = original(*x, return_aux=True)
            loss = scope['method_loss']('A02_StageRelation', old, old_aux, labels, {'quality': quality})
            loss.backward()
            torch.manual_seed(91)
            new, aux = minimal(*x, return_aux=True)
            loss, _ = training_loss(new, aux, labels, quality); loss.backward()
            for (name, old_p), (_, new_p) in zip(original.named_parameters(), minimal.named_parameters()):
                if old_p.grad is not None:
                    torch.testing.assert_close(old_p.grad, new_p.grad, rtol=0, atol=0, msg=name)
            checks.append('tiny_training_gradients_exact')
        checks.extend([backbone + '_state_keys_compatible', backbone + '_eval_logits_exact', backbone + '_loss_exact'])
        del original, minimal
    args.output.write_text(json.dumps({'passed': checks}, indent=2), encoding='utf-8')
    print(checks)


if __name__ == '__main__':
    main()
