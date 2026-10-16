"""PyTorch: backprop correctness, why non-linearity matters, and the MLP estimator."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from ml import synthetic  # noqa: E402
from ml.deep_learning import autograd_demo as ad  # noqa: E402
from ml.deep_learning.mlp import MLP, TabularDataset, TorchMLPClassifier, resolve_device  # noqa: E402
from ml.deep_learning.train import compare, training_curve  # noqa: E402
from ml.features import build_feature_table  # noqa: E402


@pytest.fixture()
def data():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(200, 4, generator=g)
    true_w = torch.tensor([2.0, -1.0, 0.5, 0.0])
    y = (torch.rand(200, generator=g) < torch.sigmoid(X @ true_w + 0.3)).float()
    return X, y


class TestBackprop:
    def test_autograd_matches_the_chain_rule(self, data):
        X, y = data
        w, b = torch.randn(4), torch.tensor(0.1)

        gw_manual, gb_manual = ad.manual_gradients(X, y, w, b)
        gw_auto, gb_auto = ad.autograd_gradients(X, y, w, b)

        torch.testing.assert_close(gw_auto, gw_manual)
        torch.testing.assert_close(gb_auto, gb_manual)

    def test_autograd_matches_finite_differences(self, data):
        """∂L/∂w_j ≈ (L(w + εe_j) - L(w - εe_j)) / 2ε: the definition of a derivative."""
        X, y = data
        X, y = X.double(), y.double()
        w, b = torch.randn(4, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
        grad, _ = ad.autograd_gradients(X, y, w, b)

        eps = 1e-6
        numeric = torch.stack([
            (ad.logistic_loss(X, y, w + eps * e, b) - ad.logistic_loss(X, y, w - eps * e, b)) / (2 * eps)
            for e in torch.eye(4, dtype=torch.float64)
        ])
        torch.testing.assert_close(grad, numeric, rtol=1e-5, atol=1e-8)

    def test_gradient_descent_learns_what_sklearn_learns(self, data):
        X, y = data
        w, b, history = ad.gradient_descent(X, y, learning_rate=0.5, steps=2000)

        assert history[-1] < history[0]
        # Convex loss + small step: never increases, up to float32 rounding (~1e-7 at this scale).
        assert all(later <= earlier + 1e-6 for earlier, later in zip(history, history[1:], strict=False))
        reference = LogisticRegression(penalty=None, max_iter=5000).fit(X.numpy(), y.numpy())
        np.testing.assert_allclose(w.numpy(), reference.coef_[0], atol=0.05)
        assert float(b) == pytest.approx(reference.intercept_[0], abs=0.05)

    def test_stable_loss_for_extreme_logits(self):
        X = torch.tensor([[1000.0], [-1000.0]])
        loss = ad.logistic_loss(X, torch.tensor([1.0, 0.0]), torch.tensor([1.0]), torch.tensor(0.0))
        assert torch.isfinite(loss) and float(loss) < 1e-6


class TestNetwork:
    def test_shapes_and_parameter_count(self):
        net = MLP(17, hidden=(64, 32))
        out = net(torch.randn(8, 17))

        assert out.shape == (8,)
        # (17·64 + 64) + (64·32 + 32) + (32·1 + 1)
        assert sum(p.numel() for p in net.parameters()) == 1152 + 2080 + 33

    def test_stacked_linear_layers_without_activation_are_one_linear_map(self):
        """Why ReLU exists: W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)."""
        l1, l2 = torch.nn.Linear(3, 5), torch.nn.Linear(5, 2)
        x = torch.randn(10, 3)
        W = l2.weight @ l1.weight
        b = l2.weight @ l1.bias + l2.bias
        torch.testing.assert_close(l2(l1(x)), x @ W.T + b)

    def test_xor_needs_a_non_linear_model(self):
        """XOR is not linearly separable: logistic regression is stuck at chance, an MLP is not."""
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(800, 2)).astype(np.float32)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

        linear = LogisticRegression().fit(X, y).score(X, y)
        mlp = TorchMLPClassifier(hidden=(16, 16), dropout=0.0, learning_rate=0.01, max_epochs=300,
                                 patience=50, batch_size=64).fit(X, y)

        assert linear < 0.65
        assert (mlp.predict(X) == y).mean() > 0.95

    def test_dataset(self):
        ds = TabularDataset(np.ones((3, 2)), np.array([0, 1, 1]))
        x, y = ds[1]
        assert len(ds) == 3 and x.dtype == torch.float32 and float(y) == 1


class TestEstimator:
    @pytest.fixture(scope="class")
    def toy(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(600, 5)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 600) > 0).astype(int)
        return X, y

    def test_sklearn_contract(self, toy):
        X, y = toy
        model = TorchMLPClassifier(max_epochs=20)
        assert clone(model).get_params()["max_epochs"] == 20

        proba = model.fit(X, y).predict_proba(X)
        assert proba.shape == (600, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-6)
        assert set(model.predict(X)) <= {0, 1}

    def test_deterministic_with_a_seed(self, toy):
        X, y = toy
        a = TorchMLPClassifier(max_epochs=15, random_state=3).fit(X, y).predict_proba(X)
        b = TorchMLPClassifier(max_epochs=15, random_state=3).fit(X, y).predict_proba(X)
        np.testing.assert_allclose(a, b)

    def test_early_stopping_restores_the_best_epoch(self, toy):
        X, y = toy
        model = TorchMLPClassifier(hidden=(128, 128), dropout=0.0, learning_rate=0.01, max_epochs=200,
                                   patience=5).fit(X, y)

        losses = [h["val_loss"] for h in model.history_]
        assert model.n_epochs_ < 200                          # stopped early
        assert model.n_epochs_ - model.best_epoch_ == 5       # after `patience` non-improving epochs
        assert losses[model.best_epoch_ - 1] == pytest.approx(min(losses), abs=1e-5)

    def test_rejects_missing_values(self, toy):
        X, y = toy
        X = X.copy()
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="imputed"):
            TorchMLPClassifier(max_epochs=2).fit(X, y)

    def test_device_resolution(self, monkeypatch):
        assert resolve_device("cpu").type == "cpu"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("auto").type == "cpu"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto").type == "cuda"


def test_comparison_end_to_end():
    logs, rooms = synthetic.generate(n_users=20, days=40, seed=2)
    report = compare(build_feature_table(logs, rooms), "synthetic", n_splits=2, max_epochs=15)

    assert set(report.cv) == {"logistic_regression", "random_forest_tuned", "mlp"}
    assert report.best_classical in {"logistic_regression", "random_forest_tuned"}
    assert "SYNTHETIC DATA" in report.to_text()
    assert report.mlp_info["parameters"] > 0
    fig = training_curve(report.history, report.mlp_info["best_epoch"])
    assert len(fig.axes[0].lines) == 3  # train, validation, best-epoch marker
