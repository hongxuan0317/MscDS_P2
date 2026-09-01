"""Compare the original and expanded matchers without fitting any model."""

from __future__ import annotations

import itertools
import json

import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, INCIDENT_MODEL_PATH, INCIDENT_PREPROCESSOR_PATH
from utils import create_scenario_vector, neutral_context_values

CONTEXT_VARIABLES = [
    "median_income", "poverty_rate", "unemployment_rate", "gini_index",
    "bachelor_plus_rate", "less_than_highschool_rate", "population_density", "land_area_sqmi",
]

model = joblib.load(INCIDENT_MODEL_PATH)
preprocessor = joblib.load(INCIDENT_PREPROCESSOR_PATH)
dictionary = pd.read_csv(DATA_DIR / "incident_feature_dictionary.csv")
reference = np.load(DATA_DIR / "incident_modelling_space_mean.npy")
quantiles = pd.read_csv(DATA_DIR / "context_reference_values.csv")
levels = preprocessor["category_levels"]


def value(variable: str, label: str) -> float:
    row = quantiles[(quantiles["variable"] == variable) & (quantiles["selection_label"] == label)]
    return float(row.iloc[0]["raw_value"])


def percentile_context(level: str, density_level: str, land_level: str | None = None) -> dict[str, float]:
    standard_label = {
        "lower": "Lower (25th percentile)",
        "typical": "Typical (median)",
        "higher": "Higher (75th percentile)",
    }[level]
    context = {
        variable: value(variable, standard_label)
        for variable in CONTEXT_VARIABLES if variable != "population_density"
    }
    context["population_density"] = value("population_density", density_level)
    if land_level:
        context["land_area_sqmi"] = value("land_area_sqmi", land_level)
    return context


neutral = neutral_context_values(preprocessor, CONTEXT_VARIABLES)
expanded_contexts = {
    "Neutral reference": neutral,
    "Lower / rural": percentile_context("lower", "Rural"),
    "Typical / mixed": percentile_context("typical", "Mixed"),
    "Higher / urban": percentile_context("higher", "Urban"),
    "Lower / highly urban": percentile_context("lower", "Highly urban", "Lower (25th percentile)"),
    "Higher / rural": percentile_context("higher", "Rural", "Higher (75th percentile)"),
    "Typical / highly urban": percentile_context("typical", "Highly urban", "Lower (25th percentile)"),
}

situational_grid = itertools.product(
    levels["armed_with"], levels["threat_type"], levels["flee_status"], [False, True]
)
situational_cases = list(situational_grid)
rows = []


def record(version: str, context_name: str, context: dict[str, float] | None, case: tuple) -> None:
    armed, threat, flee, body = case
    vector = create_scenario_vector(
        dictionary, reference, armed, threat, flee, body,
        context_values=context, preprocessor=preprocessor if context else None,
    )
    distances = np.linalg.norm(model.cluster_centers_ - vector.reshape(1, -1), axis=1)
    order = np.argsort(distances)
    row = {
        "matcher": version,
        "context_scenario": context_name,
        "armed_with": armed,
        "threat_type": threat,
        "flee_status": flee,
        "body_camera": body,
        "closest_cluster": int(order[0]),
        "first_distance": float(distances[order[0]]),
        "second_distance": float(distances[order[1]]),
        "first_vs_second_margin": float(distances[order[1]] - distances[order[0]]),
    }
    row.update({f"distance_cluster_{cluster}": float(distance) for cluster, distance in enumerate(distances)})
    rows.append(row)


for case in situational_cases:
    record("Original four-input matcher", "Neutral reference", None, case)
for context_name, context in expanded_contexts.items():
    for case in situational_cases:
        record("Expanded matcher", context_name, context, case)

scenario_results = pd.DataFrame(rows)
scenario_results.to_csv(DATA_DIR / "matcher_responsiveness_scenarios.csv", index=False)

summary = (
    scenario_results.groupby(["matcher", "closest_cluster"]).size().rename("scenarios").reset_index()
)
totals = scenario_results.groupby("matcher").size().rename("total_scenarios")
summary = summary.merge(totals, on="matcher")
summary["percentage"] = 100 * summary["scenarios"] / summary["total_scenarios"]
summary.to_csv(DATA_DIR / "matcher_responsiveness_summary.csv", index=False)

expanded_rows = scenario_results[scenario_results["matcher"] == "Expanded matcher"]
example_indices = expanded_rows.groupby("closest_cluster")["first_vs_second_margin"].idxmax()
examples = expanded_rows.loc[example_indices, [
    "closest_cluster", "context_scenario", "armed_with", "threat_type", "flee_status",
    "body_camera", "first_distance", "second_distance", "first_vs_second_margin",
]].sort_values("closest_cluster")
examples.to_csv(DATA_DIR / "matcher_responsiveness_examples.csv", index=False)

comparison = {}
for matcher, frame in scenario_results.groupby("matcher"):
    distribution = frame["closest_cluster"].value_counts(normalize=True).sort_index()
    entropy = float(-(distribution * np.log2(distribution)).sum())
    comparison[matcher] = {
        "scenarios": int(len(frame)),
        "clusters_reached": int(distribution.size),
        "largest_cluster_share_pct": float(100 * distribution.max()),
        "distribution_entropy_bits": entropy,
        "mean_first_vs_second_margin": float(frame["first_vs_second_margin"].mean()),
        "cluster_distribution_pct": {str(k): float(100 * v) for k, v in distribution.items()},
    }

baseline = comparison["Original four-input matcher"]
expanded = comparison["Expanded matcher"]
material_improvement = (
    expanded["clusters_reached"] > baseline["clusters_reached"]
    or expanded["distribution_entropy_bits"] > baseline["distribution_entropy_bits"] + 0.10
    or expanded["largest_cluster_share_pct"] < baseline["largest_cluster_share_pct"] - 10
)
dominates = expanded["largest_cluster_share_pct"] >= 80
all_profiles = expanded["clusters_reached"] == 4
retain = material_improvement and expanded["clusters_reached"] >= 3 and not dominates

audit = {
    "design": {
        "situational_combinations": len(situational_cases),
        "expanded_context_scenarios": list(expanded_contexts),
        "model_refitted": False,
    },
    "comparison": comparison,
    "representative_examples": examples.to_dict(orient="records"),
    "answers": {
        "context_materially_improved_variation": bool(material_improvement),
        "one_cluster_dominates_at_least_80_percent": bool(dominates),
        "all_four_profiles_reached": bool(all_profiles),
        "expanded_matcher_useful_enough_to_retain": bool(retain),
    },
}
(DATA_DIR / "matcher_responsiveness_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(summary.to_string(index=False))
print(json.dumps(audit, indent=2))
