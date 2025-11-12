from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.semantic_tagger import SemanticThresholds, generate_semantic_tags


def test_high_stability_and_low_hazard_tags():
    payload = {
        "components": {"coherence": 0.85, "stability": 0.82, "entropy": 0.3},
        "hazard": 0.05,
        "coherence_delta": 0.03,
        "lambda_slope": -0.1,
    }
    tags = set(generate_semantic_tags(payload))
    assert "highly_stable" in tags
    assert "low_hazard_environment" in tags
    assert "strengthening_structure" in tags
    assert "improving_stability" in tags


def test_threshold_overrides_take_precedence():
    payload = {
        "components": {"coherence": 0.65, "stability": 0.65, "entropy": 0.2},
        "hazard": 0.2,
    }
    overrides = {"high_coherence": 0.6, "high_stability": 0.6, "low_hazard": 0.3}
    tags = set(generate_semantic_tags(payload, overrides=overrides))
    assert "highly_stable" in tags
    assert "low_hazard_environment" in tags


def test_entropy_fallback_from_coherence_distance():
    payload = {"components": {"coherence": 0.05}, "rupture": 0.5}
    thresholds = SemanticThresholds(high_entropy=0.9, high_rupture=0.4)
    tags = set(generate_semantic_tags(payload, thresholds=thresholds))
    assert "chaotic_price_action" in tags
    assert "high_rupture_event" in tags


def test_volatility_crush_detection():
    payload = {
        "components": {"coherence": 0.8, "stability": 0.82},
        "metrics": {"volatility_ratio": 0.4, "price_zscore": 0.5},
    }
    tags = set(generate_semantic_tags(payload))
    assert "volatility_crush" in tags


def test_structural_regime_shift_detection():
    payload = {
        "components": {"coherence": 0.85, "stability": 0.8},
        "lambda_slope": 0.08,
        "coherence_delta": 0.09,
    }
    tags = set(generate_semantic_tags(payload))
    assert "structural_regime_shift" in tags


def test_momentum_breakout_tag():
    payload = {"metrics": {"momentum_short": 0.001}}
    thresholds = SemanticThresholds(momentum_breakout=0.0005)
    tags = set(generate_semantic_tags(payload, thresholds=thresholds))
    assert "momentum_breakout" in tags


def test_price_breakout_tag():
    payload = {"metrics": {"price_change_short": 0.001}}
    thresholds = SemanticThresholds(price_breakout=0.0004)
    tags = set(generate_semantic_tags(payload, thresholds=thresholds))
    assert "price_breakout" in tags
