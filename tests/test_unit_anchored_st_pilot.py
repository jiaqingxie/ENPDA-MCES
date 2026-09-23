import torch

from nema.models.nema import NEMAModel
from scripts.run_unit_anchored_st_pilot import checkpoint_score, reset_metric_to_unit


def test_reset_metric_is_exactly_unit_for_arbitrary_features():
    model = NEMAModel(hidden_dim=8)
    with torch.no_grad():
        for parameter in model.metric.parameters():
            parameter.add_(torch.randn_like(parameter))
    reset_metric_to_unit(model, seed=17)
    features = torch.randn(2, 3, 11)
    metric = model.metric(features)
    assert torch.equal(metric, torch.ones_like(metric))
    assert all(parameter.requires_grad for parameter in model.metric.parameters())
    assert not model.initial_weights.requires_grad
    assert not model.initial_bias.requires_grad
    assert not model.step_logits.requires_grad


def test_checkpoint_score_prefers_hard_advantage_then_earlier_epoch():
    baseline = {"epoch": 0, "aggregate_normalized_advantage": 0.0}
    positive = {"epoch": 3, "aggregate_normalized_advantage": 0.01}
    later_tie = {"epoch": 8, "aggregate_normalized_advantage": 0.01}
    assert checkpoint_score(positive) > checkpoint_score(baseline)
    assert checkpoint_score(positive) > checkpoint_score(later_tie)
