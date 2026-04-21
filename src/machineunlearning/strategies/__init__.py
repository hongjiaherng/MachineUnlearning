from machineunlearning.strategies import (
    amnesiac,
    bad_teacher,
    baseline,
    boundary,
    fine_tune,
    fisher,
    gradient_ascent,
    ntk,
    retrain,
    scrub,
    ssd,
    unsir,
)
from machineunlearning.strategies.base import UnlearnContext

STRATEGY_REGISTRY = {
    "baseline": baseline.unlearn,
    "retrain": retrain.unlearn,
    "fine_tune": fine_tune.unlearn,
    "gradient_ascent": gradient_ascent.unlearn,
    "bad_teacher": bad_teacher.unlearn,
    "scrub": scrub.unlearn,
    "amnesiac": amnesiac.unlearn,
    "boundary": boundary.unlearn,
    "ntk": ntk.unlearn,
    "fisher": fisher.unlearn,
    "unsir": unsir.unlearn,
    "ssd": ssd.unlearn,
}

__all__ = ["STRATEGY_REGISTRY", "UnlearnContext"]
