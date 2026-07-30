#!/usr/bin/env python3
"""Isolated full-data RF validation search without OOD/test access.

The existing full-train min_samples_leaf=5 model is audited and reused
read-only. New min_samples_leaf=20 and 50 forests are fitted on the same 29,277
formal training conditions and compared on the same 3,252 formal validation
conditions. Existing RF results, models and selection are never modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baseline_common import (
    RUN,
    fit_train_only_condition_encoder,
    load_training_cache,
    regression_metrics,
    require,
    select_train_only_control_features,
    sha256,
    write_json_atomic,
)
from run_mlp_search import hash_strings
from run_rf_search import (
    dump_joblib_atomic,
    make_forest,
    model_parameters,
    resource_snapshot,
    write_csv_atomic,
)


NEW_LEAF_SIZES = (20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUN / "sensitivity/rf_full_data_validation_search_20260727",
    )
    parser.add_argument("--contract", type=Path, default=RUN / "config/search_contract.json")
    parser.add_argument("--formal-root", type=Path, default=RUN)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def fixed_parameters(rf_contract: dict, leaf: int, smoke_test: bool) -> dict:
    parameters = model_parameters(rf_contract, leaf, smoke_test)
    if not smoke_test:
        require(
            parameters
            == {
                "n_estimators": 300,
                "min_samples_leaf": int(leaf),
                "max_depth": 10,
                "max_features": "sqrt",
                "bootstrap": True,
                "n_jobs": 24,
                "random_state": 42,
            },
            f"formal RF parameters changed for leaf={leaf}",
        )
    return parameters


def result_row(
    *,
    leaf: int,
    parameters: dict,
    metrics: dict,
    model: Path,
    model_hash: str,
    source: str,
    fit_seconds: float,
) -> dict:
    return {
        "min_samples_leaf": int(leaf),
        "n_estimators": int(parameters["n_estimators"]),
        "max_depth": int(parameters["max_depth"]),
        "max_features": str(parameters["max_features"]),
        "bootstrap": bool(parameters["bootstrap"]),
        "n_jobs": int(parameters["n_jobs"]),
        "random_state": int(parameters["random_state"]),
        "formal_validation_delta_mse": float(metrics["delta_mse"]),
        "formal_validation_median_condition_delta_pearson": float(
            metrics["median_condition_delta_pearson"]
        ),
        "formal_validation_mean_condition_absolute_r2": float(
            metrics["mean_condition_absolute_r2"]
        ),
        "formal_validation_mean_condition_absolute_pearson": float(
            metrics["mean_condition_absolute_pearson"]
        ),
        "model": str(model),
        "model_sha256": model_hash,
        "source": source,
        "fit_seconds": float(fit_seconds),
    }


def audit_formal_features(cache, formal_root: Path, design_path: Path) -> dict:
    input_hashes_path = formal_root / "provenance/rf/input_hashes_and_stats.json"
    input_hashes = json.loads(input_hashes_path.read_text())
    current_files = {
        "cache_manifest": cache.root / "cache_manifest.json",
        "training_metadata": cache.root / "train/metadata.csv",
        "genes": cache.root / "genes.csv",
        "rf_design": design_path,
    }
    for key, path in current_files.items():
        require(
            input_hashes[key]["sha256"] == sha256(path),
            f"formal leaf=5 provenance differs for {key}",
        )

    condition_features, encoder = fit_train_only_condition_encoder(cache)
    formal_encoder_path = formal_root / "provenance/rf/condition_encoder.json"
    require(
        json.loads(formal_encoder_path.read_text()) == encoder,
        "formal leaf=5 condition encoder differs",
    )
    selected_indices, selected_variances = select_train_only_control_features(
        cache, 1_500
    )
    formal_features_path = (
        formal_root / "provenance/rf/selected_train_only_control_features.csv"
    )
    formal_features = pd.read_csv(formal_features_path)
    require(len(formal_features) == 1_500, "formal RF feature count differs")
    require(
        np.array_equal(
            formal_features["gene_index_zero_based"].to_numpy(dtype=np.int64),
            selected_indices.astype(np.int64),
        ),
        "formal RF selected gene indices differ",
    )
    require(
        formal_features["gene"].astype(str).tolist()
        == cache.genes[selected_indices].astype(str).tolist(),
        "formal RF selected gene names differ",
    )
    require(
        np.allclose(
            formal_features[
                "weighted_formal_train_control_variance"
            ].to_numpy(dtype=np.float64),
            selected_variances.astype(np.float64),
            rtol=1e-12,
            atol=1e-15,
        ),
        "formal RF selected-feature variances differ",
    )
    return {
        "status": "FORMAL_FEATURES_MATCH",
        "condition_feature_count": int(condition_features.shape[1]),
        "selected_control_feature_count": 1_500,
        "rf_input_feature_count": int(condition_features.shape[1] + 1_500),
        "condition_encoder_sha256": sha256(formal_encoder_path),
        "selected_features_sha256": sha256(formal_features_path),
        "rf_design_sha256": sha256(design_path),
    }


def audit_and_recompute_leaf5(
    *,
    formal_root: Path,
    rf_contract: dict,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_controls: np.ndarray,
    train_hash: str,
    validation_hash: str,
    input_dim: int,
) -> tuple[dict, dict]:
    final_fit_path = formal_root / "results/rf/final_fit.json"
    metrics_path = formal_root / "results/rf/formal_validation_metrics.json"
    selection_path = formal_root / "results/rf/selection.json"
    model_path = formal_root / "models/rf/final_random_forest.joblib"
    final_fit = json.loads(final_fit_path.read_text())
    reported = json.loads(metrics_path.read_text())
    selection = json.loads(selection_path.read_text())
    parameters = fixed_parameters(rf_contract, 5, False)

    require(final_fit["status"] == "FINAL_FIT_OK", "formal leaf=5 fit did not pass")
    require(reported["status"] == "FORMAL_VALIDATION_DIAGNOSTIC_OK", "formal leaf=5 validation did not pass")
    require(selection["status"] == "SELECTION_FROZEN", "formal RF selection is not frozen")
    require(final_fit["model_parameters"] == parameters, "formal leaf=5 fit parameters differ")
    require(reported["model_parameters"] == parameters, "formal leaf=5 metric parameters differ")
    require(final_fit["formal_train_conditions"] == 29_277, "formal leaf=5 train count differs")
    require(final_fit["formal_validation_conditions"] == 3_252, "formal leaf=5 validation count differs")
    require(reported["formal_train_conditions"] == 29_277, "reported leaf=5 train count differs")
    require(reported["formal_validation_conditions"] == 3_252, "reported leaf=5 validation count differs")
    model_hash = sha256(model_path)
    require(final_fit["model_sha256"] == model_hash, "formal leaf=5 model hash differs")
    require(reported["model_sha256"] == model_hash, "formal leaf=5 metric model hash differs")
    require(selection["final_model_sha256"] == model_hash, "formal leaf=5 selection model hash differs")
    require(final_fit["formal_validation_metrics_sha256"] == sha256(metrics_path), "formal leaf=5 metric file hash differs")
    require(selection["test_response_accessed"] is False, "formal RF accessed test response")

    model = joblib.load(model_path)
    require(model.get_params()["n_estimators"] == 300, "loaded leaf=5 tree count differs")
    require(model.get_params()["min_samples_leaf"] == 5, "loaded leaf=5 size differs")
    require(model.get_params()["max_depth"] == 10, "loaded leaf=5 depth differs")
    require(model.get_params()["max_features"] == "sqrt", "loaded leaf=5 max_features differs")
    require(model.get_params()["bootstrap"] is True, "loaded leaf=5 bootstrap differs")
    require(model.get_params()["random_state"] == 42, "loaded leaf=5 random state differs")
    require(model.n_features_in_ == input_dim, "loaded leaf=5 input feature count differs")
    require(len(model.estimators_) == 300, "loaded leaf=5 fitted tree count differs")
    prediction = np.asarray(model.predict(x_valid), dtype=np.float32)
    require(prediction.shape == y_valid.shape, "leaf=5 validation prediction shape differs")
    metrics = regression_metrics(y_valid, prediction, valid_controls)
    require(
        np.isclose(
            metrics["delta_mse"],
            reported["metrics"]["delta_mse"],
            rtol=1e-10,
            atol=1e-12,
        ),
        "recomputed leaf=5 validation MSE differs",
    )
    del model, prediction
    gc.collect()
    row = result_row(
        leaf=5,
        parameters=parameters,
        metrics=metrics,
        model=model_path,
        model_hash=model_hash,
        source="read-only reused formal model after prediction recomputation",
        fit_seconds=float(final_fit["fit_seconds"]),
    )
    audit = {
        "status": "LEAF5_REUSE_AUDIT_OK",
        "formal_train_conditions": 29_277,
        "formal_validation_conditions": 3_252,
        "formal_train_ids_sha256": train_hash,
        "formal_validation_ids_sha256": validation_hash,
        "model_parameters": parameters,
        "model": str(model_path),
        "model_sha256": model_hash,
        "reported_validation_delta_mse": float(reported["metrics"]["delta_mse"]),
        "recomputed_validation_delta_mse": float(metrics["delta_mse"]),
        "metric_recomputed_on_current_aligned_formal_validation": True,
        "test_response_accessed": False,
    }
    return row, audit


def fit_or_resume_candidate(
    *,
    candidate_root: Path,
    leaf: int,
    parameters: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_controls: np.ndarray,
    provenance_hash: str,
    smoke_test: bool,
) -> dict:
    candidate_root.mkdir(parents=True, exist_ok=True)
    model_path = candidate_root / "random_forest.joblib"
    artifact_path = candidate_root / "model_artifact.json"
    metrics_path = candidate_root / "formal_validation_metrics.json"
    result_path = candidate_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        require(result["status"] == "CANDIDATE_OK", f"leaf={leaf} result did not pass")
        require(result["provenance_sha256"] == provenance_hash, f"leaf={leaf} provenance changed")
        require(result["model_parameters"] == parameters, f"leaf={leaf} parameters changed")
        require(model_path.is_file() and sha256(model_path) == result["model_sha256"], f"leaf={leaf} model changed")
        require(metrics_path.is_file() and sha256(metrics_path) == result["metrics_sha256"], f"leaf={leaf} metrics changed")
        return result["comparison_row"]

    if artifact_path.is_file():
        artifact = json.loads(artifact_path.read_text())
        require(artifact["status"] == "MODEL_SAVED", f"leaf={leaf} artifact incomplete")
        require(artifact["provenance_sha256"] == provenance_hash, f"leaf={leaf} artifact provenance changed")
        require(artifact["model_parameters"] == parameters, f"leaf={leaf} artifact parameters changed")
        require(model_path.is_file() and sha256(model_path) == artifact["model_sha256"], f"leaf={leaf} saved model changed")
        fit_seconds = float(artifact["fit_seconds"])
    else:
        model = make_forest(parameters)
        started = time.monotonic()
        model.fit(x_train, y_train)
        fit_seconds = time.monotonic() - started
        dump_joblib_atomic(model_path, model)
        del model
        gc.collect()
        artifact = {
            "status": "MODEL_SAVED",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "fit_seconds": float(fit_seconds),
            "fit_scope": "all formal internal-training conditions" if not smoke_test else "all smoke training conditions",
            "fit_conditions": int(len(x_train)),
            "model_parameters": parameters,
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "provenance_sha256": provenance_hash,
            "test_response_accessed": False,
            "smoke_test": bool(smoke_test),
        }
        write_json_atomic(artifact_path, artifact)

    model = joblib.load(model_path)
    prediction = np.asarray(model.predict(x_valid), dtype=np.float32)
    require(prediction.shape == y_valid.shape, f"leaf={leaf} prediction shape differs")
    require(np.isfinite(prediction).all(), f"leaf={leaf} prediction is non-finite")
    metrics = regression_metrics(y_valid, prediction, valid_controls)
    del model, prediction
    gc.collect()
    metrics_payload = {
        "status": "FORMAL_VALIDATION_OK" if not smoke_test else "SMOKE_VALIDATION_OK",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "model_parameters": parameters,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "training_conditions": int(len(x_train)),
        "validation_conditions": int(len(x_valid)),
        "provenance_sha256": provenance_hash,
        "test_response_accessed": False,
        "smoke_test": bool(smoke_test),
    }
    write_json_atomic(metrics_path, metrics_payload)
    row = result_row(
        leaf=leaf,
        parameters=parameters,
        metrics=metrics,
        model=model_path,
        model_hash=sha256(model_path),
        source="new isolated full-train candidate",
        fit_seconds=fit_seconds,
    )
    result = {
        "status": "CANDIDATE_OK",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_parameters": parameters,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "provenance_sha256": provenance_hash,
        "comparison_row": row,
        "test_response_accessed": False,
        "smoke_test": bool(smoke_test),
        "resource_final": resource_snapshot(),
    }
    write_json_atomic(result_path, result)
    return row


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    formal_root = args.formal_root.resolve()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    rf_contract = contract["random_forest"]
    require(int(rf_contract["n_estimators"]) == 300, "tree count changed")
    require(rf_contract["max_depth"] == 10, "max depth changed")
    require(rf_contract["max_features"] == "sqrt", "max_features changed")
    require(rf_contract["bootstrap"] is True, "bootstrap changed")
    require(int(rf_contract["n_jobs"]) == 24, "n_jobs changed")
    require(int(rf_contract["random_state"]) == 42, "random_state changed")

    cache = load_training_cache(cache_root)
    require(
        set(cache.metadata["search_split"].astype(str).unique()) == {"train", "valid"},
        "cache contains a split other than formal train/validation",
    )
    train_indices = np.flatnonzero(cache.train_mask)
    validation_indices = np.flatnonzero(cache.valid_mask)
    require(set(train_indices).isdisjoint(validation_indices), "formal split overlap")
    if not args.smoke_test:
        require(len(train_indices) == 29_277, "formal train count differs")
        require(len(validation_indices) == 3_252, "formal validation count differs")

    condition_ids = cache.metadata["condition_id"].astype(str)
    train_hash = hash_strings(condition_ids.iloc[train_indices])
    validation_hash = hash_strings(condition_ids.iloc[validation_indices])
    delta_path = cache_root / "model_inputs/delta_expression.npy"
    design_path = cache_root / "model_inputs/rf_design.npy"
    require(delta_path.is_file() and design_path.is_file(), "precomputed RF inputs missing")
    delta = np.load(delta_path, mmap_mode="r")
    design = np.load(design_path, mmap_mode="r")
    require(delta.shape == cache.responses.shape, "delta shape differs")
    require(design.shape[0] == len(cache.metadata), "RF design row count differs")
    if not args.smoke_test:
        require(design.shape[1] == 1_927, "formal RF input dimension differs")

    output_root.mkdir(parents=True, exist_ok=True)
    provenance_root = output_root / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    feature_audit = None
    formal_selection_hash = None
    formal_model_hash = None
    if not args.smoke_test:
        feature_audit = audit_formal_features(cache, formal_root, design_path)
        formal_selection_hash = sha256(formal_root / "results/rf/selection.json")
        formal_model_hash = sha256(formal_root / "models/rf/final_random_forest.joblib")

    provenance = {
        "status": "RF_FULL_DATA_VALIDATION_CONTRACT",
        "unit_of_analysis": contract["unit_of_analysis"],
        "target": contract["target"],
        "selection_metric": "formal-validation matched-control-relative delta-expression MSE",
        "formal_train_conditions": int(len(train_indices)),
        "formal_validation_conditions": int(len(validation_indices)),
        "formal_train_ids_sha256": train_hash,
        "formal_validation_ids_sha256": validation_hash,
        "training_validation_overlap": 0,
        "fixed_parameters": {
            "n_estimators": 10 if args.smoke_test else 300,
            "max_depth": 10,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": 2 if args.smoke_test else 24,
            "random_state": 42,
        },
        "min_samples_leaf_compared": [5, 20, 50] if not args.smoke_test else [20, 50],
        "new_models_fitted": [20, 50],
        "formal_leaf5_reused_read_only": not args.smoke_test,
        "feature_audit": feature_audit,
        "cache_manifest_sha256": sha256(cache_root / "cache_manifest.json"),
        "training_metadata_sha256": sha256(cache_root / "train/metadata.csv"),
        "genes_sha256": sha256(cache_root / "genes.csv"),
        "delta_expression_sha256": sha256(delta_path),
        "rf_design_sha256": sha256(design_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "test_response_accessed": False,
        "smoke_test": bool(args.smoke_test),
    }
    provenance_path = provenance_root / "analysis_contract.json"
    if provenance_path.is_file():
        require(json.loads(provenance_path.read_text()) == provenance, "RF sensitivity contract changed")
    else:
        write_json_atomic(provenance_path, provenance)
    provenance_hash = sha256(provenance_path)

    x_train = np.asarray(design[train_indices], dtype=np.float32)
    y_train = np.asarray(delta[train_indices], dtype=np.float32)
    x_valid = np.asarray(design[validation_indices], dtype=np.float32)
    y_valid = np.asarray(delta[validation_indices], dtype=np.float32)
    valid_controls = np.asarray(
        cache.controls[cache.control_index[validation_indices]], dtype=np.float32
    )
    require(x_train.shape[0] == y_train.shape[0] == len(train_indices), "train alignment failed")
    require(x_valid.shape[0] == y_valid.shape[0] == len(validation_indices), "validation alignment failed")
    require(y_train.shape[1] == y_valid.shape[1] == len(cache.genes), "gene alignment failed")

    rows = []
    if not args.smoke_test:
        leaf5_row, leaf5_audit = audit_and_recompute_leaf5(
            formal_root=formal_root,
            rf_contract=rf_contract,
            x_valid=x_valid,
            y_valid=y_valid,
            valid_controls=valid_controls,
            train_hash=train_hash,
            validation_hash=validation_hash,
            input_dim=int(design.shape[1]),
        )
        write_json_atomic(provenance_root / "leaf5_reuse_audit.json", leaf5_audit)
        rows.append(leaf5_row)

    for leaf in NEW_LEAF_SIZES:
        parameters = fixed_parameters(rf_contract, leaf, args.smoke_test)
        rows.append(
            fit_or_resume_candidate(
                candidate_root=output_root / f"min_samples_leaf_{leaf}",
                leaf=leaf,
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                valid_controls=valid_controls,
                provenance_hash=provenance_hash,
                smoke_test=args.smoke_test,
            )
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["formal_validation_delta_mse", "min_samples_leaf"], kind="stable"
    ).reset_index(drop=True)
    expected_rows = 2 if args.smoke_test else 3
    require(len(comparison) == expected_rows, "RF comparison row count differs")
    minimum = float(comparison["formal_validation_delta_mse"].min())
    require(
        int(comparison["formal_validation_delta_mse"].eq(minimum).sum()) == 1,
        "strict RF validation argmin is tied",
    )
    comparison_path = output_root / (
        "smoke_comparison.csv" if args.smoke_test else "full_data_validation_comparison.csv"
    )
    write_csv_atomic(comparison_path, comparison)
    if args.smoke_test:
        write_json_atomic(
            output_root / "smoke_checks.json",
            {
                "status": "SMOKE_OK",
                "candidate_count": 2,
                "saved_reloaded_and_validated": True,
                "test_response_accessed": False,
            },
        )
        print(json.dumps(json.loads((output_root / "smoke_checks.json").read_text()), indent=2))
        return

    require(
        sha256(formal_root / "results/rf/selection.json") == formal_selection_hash,
        "formal RF selection changed",
    )
    require(
        sha256(formal_root / "models/rf/final_random_forest.joblib") == formal_model_hash,
        "formal RF model changed",
    )
    best = comparison.iloc[0]
    summary = {
        "status": "RF_FULL_DATA_VALIDATION_SEARCH_OK",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "3252-condition formal-validation delta-expression MSE",
        "selection_mode": "strict argmin",
        "strict_argmin_min_samples_leaf": int(best["min_samples_leaf"]),
        "strict_argmin_validation_delta_mse": float(
            best["formal_validation_delta_mse"]
        ),
        "strict_argmin_model": str(best["model"]),
        "strict_argmin_model_sha256": str(best["model_sha256"]),
        "comparison": str(comparison_path),
        "comparison_sha256": sha256(comparison_path),
        "formal_selection_modified": False,
        "test_response_accessed": False,
    }
    write_json_atomic(output_root / "full_data_validation_comparison.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
