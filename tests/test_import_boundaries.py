"""Check the first import in a fresh interpreter, before pytest can load adapters."""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_clean_python(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_numerical_api_can_import_and_run_with_adapters_unavailable():
    _run_clean_python("""
        import sys
        from importlib.abc import MetaPathFinder
        from math import isclose

        excluded = {"matplotlib", "seaborn", "yfinance", "xgboost", "squarify", "statsmodels", "sklearn"}

        class BlockNonNumericalImports(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in excluded:
                    raise AssertionError(f"Unexpected import: {fullname}")

        sys.meta_path.insert(0, BlockNonNumericalImports())
        import riskoptima
        from riskoptima.options import black_scholes_price
        from riskoptima.rates import DiscountCurve
        from riskoptima.credit import merton_pd

        assert "RiskOptima" in dir(riskoptima)
        assert "riskoptima.riskoptima" not in sys.modules
        assert isclose(black_scholes_price(100, 100, 1, .05, .20), 10.450583572185565)
        assert isclose(DiscountCurve.from_zero_rates([1.], [.04]).discount_factor(1.), .9607894391523232)
        assert 0.0 < merton_pd(150., 100., .25, .04, 1.) < 1.0
        assert not (excluded & sys.modules.keys())
        """)


def test_legacy_facade_import_identity_reflection_and_pickling():
    _run_clean_python("""
        import pickle
        import riskoptima

        assert "RiskOptima" not in vars(riskoptima)
        from riskoptima import RiskOptima
        from riskoptima.riskoptima import RiskOptima as direct
        exports = {}
        exec("from riskoptima import *", exports)

        assert RiskOptima is direct is riskoptima.RiskOptima is exports["RiskOptima"]
        assert pickle.loads(pickle.dumps(RiskOptima)) is RiskOptima
        assert set(riskoptima.__all__) <= exports.keys()
        assert RiskOptima.VERSION == riskoptima.__version__

        try:
            riskoptima.missing_public_name
        except AttributeError as exc:
            assert "missing_public_name" in str(exc)
        else:
            raise AssertionError("missing names must raise AttributeError")
        """)


def test_credit_classifier_loads_its_estimator_when_requested():
    from riskoptima.credit import logistic_pd_model, predict_pd

    model = logistic_pd_model([[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1])
    low, high = predict_pd(model, [[-1.5], [1.5]])
    assert 0.0 < low < 0.5 < high < 1.0
