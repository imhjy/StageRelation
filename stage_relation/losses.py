"""Full A02 loss; quality corruption is applied only during training."""
import torch
from torch.nn import functional as F


def quality_augmentation(images, probability=.25, std=.25):
    changed, scores = [], []
    for image in images:
        corrupt = torch.rand(len(image), device=image.device) < probability
        noise = torch.randn_like(image) * std
        changed.append(image + noise * corrupt[:, None, None, None])
        scores.append(torch.where(corrupt, .1, 1.))
    return tuple(changed), torch.stack(scores, 1)


def training_loss(logits, aux, labels, quality):
    components = {
        'classification': F.cross_entropy(logits, labels),
        'shared_private': aux['losses']['shared_private'],
        'stage': .1 * F.cross_entropy(aux['stage_logits'], labels),
        'quality': .2 * F.kl_div(aux['quality_logits'].log_softmax(-1),
                                quality / quality.sum(-1, keepdim=True), reduction='batchmean'),
    }
    # Match the original summation order.
    auxiliary = logits.sum() * 0 + components['shared_private']
    auxiliary = auxiliary + components['stage'] + components['quality']
    return components['classification'] + auxiliary, components
