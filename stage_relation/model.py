"""StageRelation (A02): original encoder and full three-modality relation head."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


def mlp(source, hidden, target, dropout=.1):
    return nn.Sequential(nn.LayerNorm(source), nn.Linear(source, hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, target))


def transformer(width, layers=2, heads=8):
    layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=.1,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def present_mask(images):
    return torch.stack([image.abs().flatten(1).sum(1) > 0 for image in images], 1)


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=512, backbone='resnet50', pretrained=True, grid_size=7):
        super().__init__()
        self.backbone_name = backbone
        self.grid_size = grid_size
        if backbone == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.backbone = nn.ModuleDict({'stem': nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
                                          'layer1': model.layer1, 'layer2': model.layer2,
                                          'layer3': model.layer3, 'layer4': model.layer4})
            channels = (512, 1024, 2048)
        elif backbone == 'tiny':
            widths = (32, 64, 128, 256)
            stages = []
            source = 3
            for width in widths:
                stages.append(nn.Sequential(nn.Conv2d(source, width, 3, 2, 1, bias=False),
                                             nn.BatchNorm2d(width), nn.GELU(),
                                             nn.Conv2d(width, width, 3, 1, 1, bias=False),
                                             nn.BatchNorm2d(width), nn.GELU()))
                source = width
            self.backbone = nn.ModuleDict({'stem': stages[0], 'layer1': nn.Identity(),
                                          'layer2': stages[1], 'layer3': stages[2], 'layer4': stages[3]})
            channels = widths[1:]
        if backbone not in ('resnet50', 'tiny'):
            raise ValueError('backbone must be resnet50 or tiny')
        self.projections = nn.ModuleList(nn.Sequential(nn.Conv2d(width, feature_dim, 1, bias=False),
                                                        nn.GroupNorm(8, feature_dim), nn.GELU())
                                         for width in channels)
        self.scale_weights = nn.Parameter(torch.zeros(3))

    def stem(self, image):
        return self.backbone['stem'](image)

    def from_stem(self, stem):
        x = self.backbone['layer1'](stem)
        maps = []
        for name, projection in zip(('layer2', 'layer3', 'layer4'), self.projections):
            x = self.backbone[name](x)
            maps.append(F.adaptive_avg_pool2d(projection(x), self.grid_size))
        self.teacher_features = x.detach().mean((2,3))
        return sum(weight * feature for weight, feature in zip(self.scale_weights.softmax(0), maps))

    def forward(self, image):
        return self.from_stem(self.stem(image))


class MultiModalEncoder(nn.Module):
    def __init__(self, feature_dim=512, backbone='resnet50', pretrained=True,
                 grid_size=7, token_layers=2, token_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        first = ImageEncoder(feature_dim, backbone, pretrained, grid_size)
        self.branches = nn.ModuleList([first, copy.deepcopy(first), copy.deepcopy(first)])
        self.position = nn.Parameter(torch.randn(1, grid_size * grid_size, feature_dim) * .02)
        self.modality = nn.Parameter(torch.randn(3, 1, feature_dim) * .02)
        self.refiner = transformer(feature_dim, token_layers, token_heads)
        self.norm = nn.LayerNorm(feature_dim)
        self.frozen_backbone = False

    def freeze_backbones(self):
        self.frozen_backbone = True
        for branch in self.branches:
            branch.backbone.requires_grad_(False)
            branch.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:
            for branch in self.branches:
                branch.backbone.eval()
        return self

    def refine(self, spatial, modality):
        tokens = spatial.flatten(2).transpose(1, 2) + self.position + self.modality[modality]
        tokens = self.norm(self.refiner(tokens))
        return tokens.mean(1), tokens.transpose(1, 2).reshape_as(spatial)

    def encode_modality(self, image, modality):
        return self.refine(self.branches[modality](image), modality)

    def from_stems(self, stems):
        encoded = [self.refine(branch.from_stem(stem), i)
                   for i, (branch, stem) in enumerate(zip(self.branches, stems))]
        return torch.stack([x[0] for x in encoded], 1), torch.stack([x[1] for x in encoded], 1)

    def forward(self, images):
        encoded = [self.encode_modality(image, i) for i, image in enumerate(images)]
        return torch.stack([x[0] for x in encoded], 1), torch.stack([x[1] for x in encoded], 1)


class ProposalModel(nn.Module):
    def __init__(self, num_classes=4, feature_dim=512, backbone='resnet50', pretrained=True,
                 grid_size=7, token_layers=2, token_heads=8):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.encoder = MultiModalEncoder(feature_dim, backbone, pretrained, grid_size, token_layers, token_heads)
        self.sample_ids = None

    @property
    def encoders(self):
        return self.encoder.branches

    def set_sample_ids(self, sample_ids):
        self.sample_ids = sample_ids

    def encode(self, images):
        return self.encoder(images)

    def after_optimizer_step(self):
        pass

    def classify(self, features, maps, present):
        raise NotImplementedError

    def forward(self, slo, thickness, vf, return_aux=False):
        images = (slo, thickness, vf)
        present = present_mask(images)
        features, maps = self.encode(images)
        features = features * present[..., None]
        maps = maps * present[:, :, None, None, None]
        logits, aux = self.classify(features, maps, present)
        aux.update(features=features, present=present)
        return (logits, aux) if return_aux else logits


class StageRelation(ProposalModel):
    method_name='A02_StageRelation'
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        d=self.feature_dim
        self.stage=mlp(d*3,d,4)
        self.shared=nn.ModuleList(mlp(d,d,d) for _ in range(3))
        self.quality=mlp(d*3,d,3)
        self.experts=nn.ModuleList(mlp(d*6,d*2,d) for _ in range(4))
        self.classifier=mlp(d*7+3,d*2,self.num_classes)

    def classify(self,features,maps,present):
        shared=torch.stack([head(features[:,m]) for m,head in enumerate(self.shared)],1)
        private=features-shared
        pairs=torch.cat([torch.cat((shared[:,i]*shared[:,j],private[:,i]-private[:,j]),-1)
                         for i,j in ((0,1),(0,2),(1,2))],-1)
        stage_logits=self.stage(features.flatten(1))
        relation=(torch.stack([expert(pairs) for expert in self.experts],1)*stage_logits.softmax(-1)[...,None]).sum(1)
        quality_logits=self.quality(features.flatten(1)).masked_fill(~present,-1e4)
        quality=quality_logits.softmax(-1)
        weighted=features*(1+quality*3)[...,None]
        logits=self.classifier(torch.cat((weighted.flatten(1),private.flatten(1),relation,quality),-1))
        return logits,{'stage_logits':stage_logits,'quality_logits':quality_logits,
                       'losses':{'shared_private':.01*(F.normalize(shared,dim=-1)*F.normalize(private,dim=-1)).sum(-1).square().mean()}}
