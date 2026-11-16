import json

import pytest
import torch
from torch import nn

from paltas.config import TrainConfig
from paltas.engine import cosine_schedule, param_groups, train_one_epoch
from paltas.models import freeze_resnet_prefix


def test_vit_layer_rates_decay_without_losing_parameters_or_bias_rules():
    model = nn.Module()
    model.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
    model.pos_embed = nn.Parameter(torch.zeros(1, 2, 4))
    model.patch_embed = nn.Linear(3, 4)
    model.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    model.norm = nn.LayerNorm(4)
    model.head = nn.Linear(4, 5)
    groups = param_groups(model, .001, 10, .05, layer_decay=.5)
    assignments = {id(p): group for group in groups for p in group['params']}
    assert sum(len(g['params']) for g in groups) == len(assignments) == len(list(model.parameters()))
    assert assignments[id(model.patch_embed.weight)]['lr'] == pytest.approx(.001 * .5**2)
    assert assignments[id(model.cls_token)]['lr'] == pytest.approx(.001 * .5**2)
    assert assignments[id(model.blocks[0].weight)]['lr'] == pytest.approx(.001 * .5)
    assert assignments[id(model.blocks[1].weight)]['lr'] == pytest.approx(.001)
    assert assignments[id(model.norm.weight)]['lr'] == pytest.approx(.001)
    assert assignments[id(model.head.weight)]['lr'] == pytest.approx(.01)
    assert assignments[id(model.norm.weight)]['weight_decay'] == 0
    assert assignments[id(model.head.bias)]['weight_decay'] == 0
    assert assignments[id(model.head.weight)]['weight_decay'] == .05
    optimizer = torch.optim.AdamW(groups)
    scheduler = cosine_schedule(optimizer, total_steps=10, warmup_steps=2)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert assignments[id(model.patch_embed.weight)]['lr'] / assignments[id(model.blocks[1].weight)]['lr'] == pytest.approx(.25)


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.bn1 = nn.BatchNorm2d(4)
        for index in range(1, 5):
            setattr(self, f'layer{index}', nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.ReLU()))
        self.fc = nn.Linear(4, 5)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        for index in range(1, 5):
            x = getattr(self, f'layer{index}')(x)
        return self.fc(x.mean((2, 3)))


@pytest.mark.parametrize('stages', [2, 3])
def test_frozen_resnet_weights_and_batchnorm_stay_fixed_across_epochs(stages):
    model = SmallResNet()
    freeze_resnet_prefix(model, stages)
    prefixes = ('conv1.', 'bn1.', *(f'layer{i}.' for i in range(1, stages + 1)))
    frozen = {name: value.clone() for name, value in model.state_dict().items()
              if name.startswith(prefixes)}
    trainable_before = model.layer4[0].weight.detach().clone()
    groups = param_groups(model, .01, 1, .05)
    assert all(p.requires_grad for group in groups for p in group['params'])
    optimizer = torch.optim.AdamW(groups)
    scheduler = cosine_schedule(optimizer, 2, 0)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    batch = [(torch.randn(2, 3, 8, 8), torch.tensor([0, 1]))]
    for _ in range(2):
        train_one_epoch(model, batch, nn.CrossEntropyLoss(), optimizer, scheduler,
                        scaler, torch.device('cpu'), False, 1.0, 'test', freeze_resnet_stages=stages)
        assert model.training and not model.bn1.training and model.layer4[1].training
        assert model.layer3[1].training == (stages == 2)
        assert all(torch.equal(model.state_dict()[name], value) for name, value in frozen.items())
        assert model.conv1.weight.grad is None
    assert not torch.equal(model.layer4[0].weight, trainable_before)


@pytest.mark.parametrize('settings', [
    {'model': 'vit_base', 'layer_decay': 0}, {'model': 'vit_base', 'layer_decay': 1.1},
    {'model': 'resnet101', 'layer_decay': .75}, {'model': 'vit_base', 'freeze_resnet_stages': 2},
    {'model': 'resnet101', 'freeze_resnet_stages': 4}, {'model': 'resnet101', 'freeze_resnet_stages': 2.0}])
def test_invalid_finetuning_configs_fail_early(tmp_path, settings):
    path = tmp_path / 'invalid.yaml'
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(path)
