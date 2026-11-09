import math

import pandas as pd
import pytest
import torch

import run_feature_drift as drift
from run_daco_cifar10 import _apply_kappa_wiring, configure_training_schedule


@pytest.mark.parametrize("target,periods", [(1, 5), (17, 5), (100, 7)])
def test_cosine_pruning_budgets_sum_to_exact_target(target, periods):
    budgets = drift.cosine_pruning_budgets(target, periods)

    assert len(budgets) == periods
    assert all(value >= 0 for value in budgets)
    assert sum(budgets) == target


def test_configure_training_schedule_recomputes_period_duration():
    class OptimizerStub:
        pruning_periods = 5

    optimizer = OptimizerStub()

    configure_training_schedule(
        optimizer, warmup_epochs=10, joint_epochs=200, steps_per_epoch=391)

    assert optimizer.start_pruning_step == 3910
    assert optimizer.start_projection_step == 3910
    assert optimizer.pruning_steps == 78200
    assert optimizer.pruning_period_duration == 15640
    assert optimizer.diffusion_steps == 78200


def test_kappa_wrapper_remains_a_callable_bound_method():
    class OptimizerStub:
        param_groups = []
        bit_layers = []
        importance_score_criteria = {}

        def __init__(self):
            self.original_calls = 0

        def compute_importance_scores(self):
            self.original_calls += 1

    optimizer = OptimizerStub()

    _apply_kappa_wiring(optimizer, kappa=0.5)
    optimizer.compute_importance_scores()

    assert optimizer.original_calls == 1


def test_metric_loader_finds_full_experiment_directory_depth(tmp_path):
    run_directory = (
        tmp_path / "pruning" / "resnet20" / "fixed" / "seed_001")
    run_directory.mkdir(parents=True)
    pd.DataFrame([{"strategy": "fixed", "cka_drift_ref": 0.25}]).to_csv(
        run_directory / "metrics.csv", index=False)

    frame = drift._load_metric_files(tmp_path)

    assert frame.to_dict("records") == [
        {"strategy": "fixed", "cka_drift_ref": 0.25}]


def test_linear_cka_identical_features_has_no_drift():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(32, 11, generator=generator)

    similarity = drift.linear_cka(features, features.clone())

    assert similarity == pytest.approx(1.0, abs=1e-10)


def test_linear_cka_accepts_different_feature_dimensions():
    generator = torch.Generator().manual_seed(9)
    first = torch.randn(24, 5, generator=generator)
    second = torch.randn(24, 13, generator=generator)

    similarity = drift.linear_cka(first, second)

    assert math.isfinite(similarity)
    assert 0.0 <= similarity <= 1.0


def test_neighbor_preservation_identical_features_is_one():
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(20, 8, generator=generator)

    preservation = drift.neighbor_preservation(features, features.clone(), k=5)

    assert preservation == pytest.approx(1.0)


def test_prototype_geometry_is_invariant_to_global_feature_scale():
    generator = torch.Generator().manual_seed(13)
    features = torch.randn(30, 6, generator=generator)
    labels = torch.arange(30) % 3

    first = drift.prototype_geometry(features, labels)
    second = drift.prototype_geometry(features * 4.0, labels)

    assert drift.geometry_drift(first, second) == pytest.approx(0.0, abs=1e-6)


def test_feature_collector_accumulates_multiple_batches():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.head = torch.nn.Linear(4, 2)

        def forward(self, inputs):
            features = self.features(inputs)
            return self.head(self.pool(features).flatten(1))

    model = TinyModel()
    inputs = torch.randn(7, 3, 4, 4)
    labels = torch.arange(7) % 2
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels), batch_size=3)

    snapshot = drift.extract_snapshot(
        model, loader, torch.device("cpu"), ["features"])

    assert snapshot["features"]["features"].shape == (7, 4)
    assert snapshot["logits"].shape == (7, 2)
    assert torch.equal(snapshot["labels"], labels)
