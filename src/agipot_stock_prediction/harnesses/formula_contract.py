"""Small independent mathematical oracles; randomized properties live in tests."""
import math
from agipot_stock_prediction.formulas.ensemble import evaluate_ensemble
from agipot_stock_prediction.research.cost_model import apply_cost_to_return


class FormulaContractHarness:
    def run(self) -> dict:
        expected_net = 0.02 - 15 / 10_000
        actual_net = apply_cost_to_return(gross_return=0.02, cost_bps=15)
        result = evaluate_ensemble(skill_scores={"a": 0.2, "b": 0.5, "c": 0.8},
                                   skill_reliabilities={"a": 0.5, "b": 0.5, "c": 0.5})
        checks = {
            "basis_point_units": math.isclose(actual_net, expected_net, abs_tol=1e-12),
            "equal_reliability_mean": math.isclose(result.score, 0.5, abs_tol=1e-12),
            "normalized_weights": math.isclose(sum(result.evidence["weights"].values()), 1, abs_tol=1e-12),
            "weight_cap": max(result.evidence["weights"].values()) <= 0.4,
        }
        return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                "evidence_type": "implementation_contract", "profitability_validated": False}
