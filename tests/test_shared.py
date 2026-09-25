import numpy as np
import pytest
import torch

from backend.config import load_train_config
from backend.data.dgp import FAILURE_REASONS
from backend.models.encoder import (
    N_FIELDS,
    NUMERIC,
    FieldTokenizer,
    PaymentEncoder,
    PoolingByAttention,
    SetBlock,
)
from backend.models.features import CATEGORIES, FeatureBuilder
from backend.models.heads import outcome_loss
from backend.models.pretrain import random_field_mask
from backend.models.shared import SharedModel
from backend.models.splits import first_attempts_on, time_split
from backend.simulation.engine import simulate
from backend.simulation.interventions import apply_scenario, observe
from backend.simulation.scenarios import Scenario

VOCAB = [len(v) for v in CATEGORIES.values()]


@pytest.fixture(scope="module")
def train(dataset):
    return first_attempts_on(dataset, time_split(dataset, test_days=4).train_days)


@pytest.fixture(scope="module")
def tiny_config():
    return load_train_config().model_copy(
        update={
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            # Enough optimizer steps for pretraining to beat the frequency baseline.
            "pretrain_rows": 60_000,
            "pretrain_epochs": 2,
            "finetune_epochs": 1,
            "ensemble_size": 2,
        }
    )


@pytest.fixture(scope="module")
def shared(dataset, train, tiny_config):
    return SharedModel.fit(train, dataset.customers, dataset.merchants, tiny_config)


def test_train_config_loads():
    config = load_train_config()
    assert config.mask_rate_min <= config.mask_rate_max
    assert config.d_model % config.n_heads == 0


def test_tokenizer_uses_training_statistics(dataset, train):
    x = FeatureBuilder(dataset.customers, dataset.merchants)(train)
    tokenizer = FieldTokenizer.fit(x, n_bins=16)
    codes, numeric = tokenizer(x)
    assert codes.shape == (len(x), len(CATEGORIES))
    assert numeric.shape == (len(x), len(NUMERIC))
    assert numeric.mean(dim=0).abs().max() < 1e-3
    bins = tokenizer.numeric_bins(x)
    assert (bins >= 0).all()
    assert (bins.max(dim=0).values < torch.tensor(tokenizer.numeric_bin_counts)).all()


def test_set_blocks_and_pooling_ignore_token_order():
    torch.manual_seed(0)
    block, pool = SetBlock(16, 2, 2, 0.0).eval(), PoolingByAttention(16, 2).eval()
    h = torch.randn(4, N_FIELDS, 16)
    order = torch.randperm(N_FIELDS)
    with torch.no_grad():
        torch.testing.assert_close(pool(block(h)), pool(block(h[:, order])))


def test_masked_field_value_has_no_effect():
    torch.manual_seed(0)
    encoder = PaymentEncoder(VOCAB, 16, 2, 1, 2, 0.0).eval()
    codes = torch.stack([torch.randint(v, (8,)) for v in VOCAB], dim=1)
    numeric = torch.randn(8, len(NUMERIC))
    mask = torch.zeros(8, N_FIELDS, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, -1] = True
    other_codes, other_numeric = codes.clone(), numeric.clone()
    other_codes[:, 0] = (codes[:, 0] + 1) % VOCAB[0]
    other_numeric[:, -1] += 5.0
    with torch.no_grad():
        torch.testing.assert_close(
            encoder(codes, numeric, mask), encoder(other_codes, other_numeric, mask)
        )
        assert not torch.allclose(encoder(codes, numeric), encoder(other_codes, other_numeric))


def test_random_field_mask_rate():
    mask = random_field_mask(20_000, 0.15, 0.30, torch.Generator().manual_seed(0))
    assert mask.any(dim=1).all()
    # Mean rate 0.225 plus the one forced field per row.
    assert 0.2 < mask.float().mean().item() < 0.3


def test_outcome_loss_ignores_reason_on_success_and_uses_weights():
    torch.manual_seed(0)
    out = {
        "success_logit": torch.randn(4),
        "reason_logits": torch.randn(4, len(FAILURE_REASONS)),
        "latency_mu": torch.randn(4),
        "latency_log_sigma": torch.zeros(4),
    }
    success = torch.tensor([1.0, 1.0, 0.0, 0.0])
    latency = torch.randn(4)
    weight = torch.ones(4)
    reason = torch.tensor([-1, -1, 2, 3])
    relabelled = torch.tensor([5, 0, 2, 3])
    a = outcome_loss(out, success, reason, latency, weight, 1.0, 0.5)
    assert a == outcome_loss(out, success, relabelled, latency, weight, 1.0, 0.5)
    only_first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # The loss is a weighted mean, so scaling every weight changes nothing.
    b = outcome_loss(out, success, reason, latency, only_first, 1.0, 0.5)
    assert b == pytest.approx(outcome_loss(out, success, reason, latency, 3 * only_first, 1.0, 0.5))
    assert b != pytest.approx(a.item())


def test_pretraining_beats_frequency_baseline(shared):
    for report in shared.reports:
        assert report.pretrain.masked_accuracy > report.pretrain.frequency_baseline_accuracy
        assert report.pretrain.train_loss[-1] < report.pretrain.train_loss[0]


def test_shared_model_predicts_valid_calibrated_distributions(dataset, shared):
    test = first_attempts_on(dataset, time_split(dataset, test_days=4).test_days)
    preds = shared.predict(test)
    assert preds.p_success.shape == (len(test),)
    assert np.allclose(preds.reason_probs.sum(axis=1), 1.0)
    assert (preds.latency_median_ms > 0).all() and (preds.latency_log_sigma > 0).all()
    observed = (test["transaction_status"] == "SUCCESS").mean()
    # Successes are subsampled in training; the reweighting keeps the mean on target.
    assert preds.p_success.mean() == pytest.approx(observed, abs=0.02)
    members = shared.member_predictions(test)
    assert len(members) == 2
    assert not np.allclose(members[0].p_success, members[1].p_success)


def test_save_load_roundtrip(dataset, shared, tmp_path):
    shared.save(tmp_path)
    loaded = SharedModel.load(tmp_path, dataset.customers, dataset.merchants)
    frame = dataset.transactions.head(2000)
    np.testing.assert_array_equal(shared.predict(frame).p_success, loaded.predict(frame).p_success)
    assert loaded.train_days == shared.train_days
    assert loaded.reports == shared.reports


def test_shared_model_plugs_into_engine(dataset, shared):
    outage = {"type": "gateway_outage", "gateway": "gateway_a", "duration_minutes": 60}
    scenario = Scenario.model_validate({"interventions": [outage]})
    cf = apply_scenario(dataset, scenario, observe(dataset))
    result = simulate(dataset, cf, shared)
    assert result.source == "shared_representation"
    assert result.counterfactual.transactions == result.baseline.transactions


def test_checkpointed_members_are_resumed_not_retrained(dataset, train, tiny_config, tmp_path):
    config = tiny_config.model_copy(update={"d_model": 16, "pretrain_epochs": 0})
    fit = SharedModel.fit(
        train, dataset.customers, dataset.merchants, config, checkpoint_dir=tmp_path
    )
    frame = dataset.transactions.head(2000)
    member_1 = fit.member_predictions(frame)[1].p_success
    # Simulate an interruption after member 0, with member 0's file holding member 1's weights:
    # a resumed run must load it as saved, and retrain only the missing member.
    (tmp_path / "member_1.json").unlink()
    (tmp_path / "member_0.pt").write_bytes((tmp_path / "member_1.pt").read_bytes())
    resumed = SharedModel.fit(
        train, dataset.customers, dataset.merchants, config, checkpoint_dir=tmp_path
    )
    np.testing.assert_array_equal(resumed.member_predictions(frame)[0].p_success, member_1)
    assert (tmp_path / "member_1.json").exists()
    assert resumed.reports == fit.reports

    other = config.model_copy(update={"seed": config.seed + 1})
    with pytest.raises(ValueError, match="different config"):
        SharedModel.fit(train, dataset.customers, dataset.merchants, other, checkpoint_dir=tmp_path)
