#!/usr/bin/env python3
"""Two-stage validation search and final fit for the random-forest baseline.

The three-configuration grid is fitted only on a development subset drawn from
the formal training split. After the development-validation metric freezes one
leaf size, exactly one final forest is fitted on all formal training conditions.
The unchanged formal validation split is then used for diagnostics, not model
selection. This program never reads external OOD/test responses.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from baseline_common import (
    RUN,
    build_delta_cache,
    build_rf_design,
    fit_train_only_condition_encoder,
    load_training_cache,
    regression_metrics,
    require,
    select_train_only_control_features,
    sha256,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=RUN)
    parser.add_argument(
        "--contract", type=Path, default=RUN / "config/search_contract.json"
    )
    parser.add_argument(
        "--development-split",
        type=Path,
        default=RUN / "config/shared_development_split.csv",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def write_csv_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def dump_joblib_atomic(path: Path, value: object) -> None:
    """Write a single compressed joblib artifact and publish it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    dumped = joblib.dump(value, temporary, compress=("zlib", 1))
    require(
        len(dumped) == 1 and Path(dumped[0]) == temporary,
        "joblib did not produce the expected single temporary artifact",
    )
    temporary.replace(path)


def freeze_json(path: Path, payload: dict) -> None:
    if path.is_file():
        require(json.loads(path.read_text()) == payload, f"frozen JSON changed: {path}")
        return
    write_json_atomic(path, payload)


def freeze_csv(path: Path, table: pd.DataFrame) -> None:
    if path.is_file():
        prior = pd.read_csv(path)
        require(
            list(prior.columns) == list(table.columns) and len(prior) == len(table),
            f"frozen CSV structure changed: {path}",
        )
        for column in table.columns:
            if pd.api.types.is_numeric_dtype(table[column]):
                require(
                    np.allclose(
                        prior[column].to_numpy(dtype=np.float64),
                        table[column].to_numpy(dtype=np.float64),
                        rtol=1e-12,
                        atol=1e-15,
                    ),
                    f"frozen CSV numeric content changed in {column}: {path}",
                )
            else:
                require(
                    prior[column].astype(str).tolist()
                    == table[column].astype(str).tolist(),
                    f"frozen CSV text content changed in {column}: {path}",
                )
        return
    write_csv_atomic(path, table)


def file_stat(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
    }


def resource_snapshot() -> dict:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "process_user_seconds": float(own.ru_utime),
        "process_system_seconds": float(own.ru_stime),
        "process_peak_rss_kib": int(own.ru_maxrss),
        "children_user_seconds": float(children.ru_utime),
        "children_system_seconds": float(children.ru_stime),
        "children_peak_rss_kib": int(children.ru_maxrss),
    }


def config_name(min_samples_leaf: int) -> str:
    return f"min_samples_leaf_{min_samples_leaf}"


def make_forest(parameters: dict) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(parameters["n_estimators"]),
        criterion="squared_error",
        max_depth=parameters["max_depth"],
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=parameters["max_features"],
        bootstrap=bool(parameters["bootstrap"]),
        n_jobs=int(parameters["n_jobs"]),
        random_state=int(parameters["random_state"]),
    )


def model_parameters(
    rf_contract: dict,
    min_samples_leaf: int,
    smoke_test: bool,
) -> dict:
    return {
        "n_estimators": 10 if smoke_test else int(rf_contract["n_estimators"]),
        "min_samples_leaf": int(min_samples_leaf),
        "max_depth": rf_contract["max_depth"],
        "max_features": rf_contract["max_features"],
        "bootstrap": bool(rf_contract["bootstrap"]),
        "n_jobs": 2 if smoke_test else int(rf_contract["n_jobs"]),
        "random_state": int(rf_contract["random_state"]),
    }


def development_indices(
    cache,
    development_split_path: Path,
    seed: int,
    smoke_test: bool,
    contract: dict,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    formal_train_indices = np.flatnonzero(cache.train_mask)
    formal_train_ids = set(
        cache.metadata.loc[cache.train_mask, "condition_id"].astype(str)
    )
    formal_valid_ids = set(
        cache.metadata.loc[cache.valid_mask, "condition_id"].astype(str)
    )

    if smoke_test:
        require(
            len(formal_train_indices) >= 5,
            "smoke cache requires at least five formal-training conditions",
        )
        shuffled = np.random.default_rng(seed).permutation(formal_train_indices)
        tuning_valid_count = max(1, int(round(0.20 * len(shuffled))))
        tuning_train = np.sort(shuffled[:-tuning_valid_count])
        tuning_valid = np.sort(shuffled[-tuning_valid_count:])
        source = {
            "mode": "smoke-derived",
                "rule": (
                    "seed-42 deterministic permutation of cache formal-train rows; "
                    "80% tuning_train and 20% tuning_validation"
                ),
            "input_csv_read": False,
        }
    else:
        development_split_path = development_split_path.resolve()
        require(
            development_split_path.is_file(),
            f"shared development split is missing: {development_split_path}",
        )
        supplied = pd.read_csv(development_split_path)
        require("condition_id" in supplied.columns, "development split lacks condition_id")
        label_candidates = [
            value
            for value in ("development_split", "tuning_split")
            if value in supplied.columns
        ]
        require(
            len(label_candidates) == 1,
            "development split must contain exactly one of development_split/tuning_split",
        )
        label_column = label_candidates[0]
        condition_ids = supplied["condition_id"].astype(str)
        require(
            condition_ids.eq(condition_ids.str.strip()).all(),
            "development condition IDs contain leading/trailing whitespace",
        )
        require(condition_ids.is_unique, "development condition IDs are duplicated")
        labels = supplied[label_column].astype(str).str.strip()
        require(
            set(labels) == {"tuning_train", "tuning_validation"},
            "development split labels must be tuning_train/tuning_validation",
        )
        expected = contract["split"]["development_subset"]
        counts = labels.value_counts()
        require(
            int(counts["tuning_train"]) == int(expected["tuning_train_conditions"])
            and int(counts["tuning_validation"])
            == int(expected["tuning_validation_conditions"])
            and len(supplied) == int(expected["total_conditions"]),
            "formal development split is not exactly 4,800 tuning_train + 1,200 tuning_validation",
        )
        supplied_ids = set(condition_ids)
        require(
            supplied_ids.issubset(formal_train_ids),
            "development split includes IDs outside the formal training split",
        )
        require(
            supplied_ids.isdisjoint(formal_valid_ids),
            "development split overlaps formal validation",
        )
        row_lookup = {
            value: int(index)
            for index, value in enumerate(cache.metadata["condition_id"].astype(str))
        }
        tuning_train = np.sort(
            np.asarray(
                [
                    row_lookup[value]
                    for value in condition_ids[labels.eq("tuning_train")]
                ],
                dtype=np.int64,
            )
        )
        tuning_valid = np.sort(
            np.asarray(
                [
                    row_lookup[value]
                    for value in condition_ids[labels.eq("tuning_validation")]
                ],
                dtype=np.int64,
            )
        )
        source = {
            "mode": "shared-formal-csv",
            "input_csv_read": True,
            "input_csv": str(development_split_path),
            "input_csv_sha256": sha256(development_split_path),
            "label_column": label_column,
        }

    require(len(tuning_train) > 0 and len(tuning_valid) > 0, "empty tuning split")
    require(
        set(tuning_train).isdisjoint(tuning_valid),
        "tuning train/validation row overlap",
    )
    require(
        set(tuning_train).union(tuning_valid).issubset(set(formal_train_indices)),
        "tuning rows are not a subset of formal training",
    )
    require(
        set(tuning_train).union(tuning_valid).isdisjoint(np.flatnonzero(cache.valid_mask)),
        "tuning rows overlap formal validation",
    )
    canonical = pd.concat(
        [
            pd.DataFrame(
                {
                    "cache_row_index_zero_based": tuning_train,
                    "condition_id": cache.metadata.iloc[tuning_train][
                        "condition_id"
                    ].astype(str).to_numpy(),
                    "development_split": "tuning_train",
                }
            ),
            pd.DataFrame(
                {
                    "cache_row_index_zero_based": tuning_valid,
                    "condition_id": cache.metadata.iloc[tuning_valid][
                        "condition_id"
                    ].astype(str).to_numpy(),
                    "development_split": "tuning_validation",
                }
            ),
        ],
        ignore_index=True,
    )
    return tuning_train, tuning_valid, canonical, source


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    rf_contract = contract["random_forest"]
    seed = int(contract["seed"])
    require(int(rf_contract["n_estimators"]) == 300, "RF contract must fix 300 trees")
    require(rf_contract["max_depth"] == 10, "RF contract must fix max_depth=10")
    leaf_grid = [int(value) for value in rf_contract["min_samples_leaf"]]
    require(leaf_grid == [5, 20, 50], "RF leaf grid must be 5/20/50")

    cache = load_training_cache(args.cache_root)
    require(
        set(cache.metadata["search_split"].astype(str).unique()) == {"train", "valid"},
        "RF cache contains a split other than formal train/validation",
    )
    formal_train = np.flatnonzero(cache.train_mask)
    formal_valid = np.flatnonzero(cache.valid_mask)
    require(set(formal_train).isdisjoint(formal_valid), "formal split row overlap")
    if not args.smoke_test:
        require(
            len(formal_train) == int(contract["split"]["internal_train_conditions"]),
            "formal RF training count differs from 29,277",
        )
        require(
            len(formal_valid) == int(contract["split"]["validation_conditions"]),
            "formal RF validation count differs from 3,252",
        )

    output_root = args.output_root.resolve()
    result_root = output_root / "results/rf"
    tuning_root = result_root / "tuning"
    model_root = output_root / "models/rf"
    provenance_root = output_root / "provenance/rf"
    check_root = output_root / "checks/rf"
    for path in (result_root, tuning_root, model_root, provenance_root, check_root):
        path.mkdir(parents=True, exist_ok=True)

    tuning_train, tuning_valid, development_table, development_source = (
        development_indices(
            cache,
            args.development_split,
            seed,
            bool(args.smoke_test),
            contract,
        )
    )
    development_used_path = provenance_root / "development_split_used.csv"
    freeze_csv(development_used_path, development_table)
    development_source["canonical_split"] = str(development_used_path)
    development_source["canonical_split_sha256"] = sha256(development_used_path)
    freeze_json(provenance_root / "development_split_source.json", development_source)

    delta_path = build_delta_cache(cache)
    condition_features, encoder_contract = fit_train_only_condition_encoder(cache)
    selected_indices, selected_variances = select_train_only_control_features(
        cache, int(rf_contract["matched_control_feature_count"])
    )
    design_path = build_rf_design(cache, condition_features, selected_indices)
    delta = np.load(delta_path, mmap_mode="r")
    design = np.load(design_path, mmap_mode="r")
    require(delta.shape == cache.responses.shape, "delta target shape mismatch")
    require(design.shape[0] == len(cache.metadata), "RF design row count mismatch")
    require(
        design.shape[1] == condition_features.shape[1] + len(selected_indices),
        "RF design feature count mismatch",
    )

    selected_feature_table = pd.DataFrame(
        {
            "selection_rank_one_based": np.arange(1, len(selected_indices) + 1),
            "gene_index_zero_based": selected_indices,
            "gene": cache.genes[selected_indices],
            "weighted_formal_train_control_variance": selected_variances,
        }
    )
    feature_path = provenance_root / "selected_train_only_control_features.csv"
    encoder_path = provenance_root / "condition_encoder.json"
    freeze_csv(feature_path, selected_feature_table)
    freeze_json(encoder_path, encoder_contract)

    input_hashes = {
        "contract": {**file_stat(contract_path), "sha256": sha256(contract_path)},
        "cache_manifest": {
            **file_stat(cache.root / "cache_manifest.json"),
            "sha256": sha256(cache.root / "cache_manifest.json"),
        },
        "training_metadata": {
            **file_stat(cache.root / "train/metadata.csv"),
            "sha256": sha256(cache.root / "train/metadata.csv"),
        },
        "control_metadata": {
            **file_stat(cache.root / "control/metadata.csv"),
            "sha256": sha256(cache.root / "control/metadata.csv"),
        },
        "genes": {
            **file_stat(cache.root / "genes.csv"),
            "sha256": sha256(cache.root / "genes.csv"),
        },
        "development_split_used": {
            **file_stat(development_used_path),
            "sha256": sha256(development_used_path),
        },
        "selected_features": {**file_stat(feature_path), "sha256": sha256(feature_path)},
        "condition_encoder": {**file_stat(encoder_path), "sha256": sha256(encoder_path)},
        "delta_cache": {**file_stat(delta_path), "sha256": sha256(delta_path)},
        "rf_design": {**file_stat(design_path), "sha256": sha256(design_path)},
        "script": {**file_stat(Path(__file__)), "sha256": sha256(Path(__file__))},
    }
    input_hash_path = provenance_root / "input_hashes_and_stats.json"
    freeze_json(input_hash_path, input_hashes)
    run_fingerprint = sha256(input_hash_path)
    freeze_json(
        provenance_root / "software_environment.json",
        {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "logical_cpu_count": os.cpu_count(),
            "random_seed": seed,
        },
    )
    freeze_json(
        check_root / "alignment_checks.json",
        {
            "status": "ALIGNMENT_OK",
            "unit_of_analysis": contract["unit_of_analysis"],
            "target": contract["target"],
            "formal_train_conditions": int(len(formal_train)),
            "formal_validation_conditions": int(len(formal_valid)),
            "tuning_train_conditions": int(len(tuning_train)),
            "tuning_validation_conditions": int(len(tuning_valid)),
            "tuning_is_subset_of_formal_train": True,
            "tuning_train_validation_overlap": 0,
            "tuning_formal_validation_overlap": 0,
            "formal_train_validation_overlap": 0,
            "formal_validation_used_for_grid_selection": False,
            "formal_validation_role": "final diagnostic only",
            "condition_ids_unique": bool(cache.metadata["condition_id"].is_unique),
            "gene_count": int(len(cache.genes)),
            "gene_names_unique": bool(len(set(cache.genes)) == len(cache.genes)),
            "matched_control_rows_aligned": True,
            "selected_control_feature_count": int(len(selected_indices)),
            "feature_selection_scope": "formal internal-training conditions only",
            "condition_encoder_fit_scope": encoder_contract["fit_scope"],
            "test_response_accessed": False,
            "smoke_test": bool(args.smoke_test),
            "smoke_split_rule": development_source.get("rule"),
        },
    )

    x_tuning_train = np.asarray(design[tuning_train], dtype=np.float32)
    y_tuning_train = np.asarray(delta[tuning_train], dtype=np.float32)
    x_tuning_valid = np.asarray(design[tuning_valid], dtype=np.float32)
    y_tuning_valid = np.asarray(delta[tuning_valid], dtype=np.float32)
    tuning_valid_controls = np.asarray(
        cache.controls[cache.control_index[tuning_valid]], dtype=np.float32
    )
    require(
        x_tuning_train.shape[0] == y_tuning_train.shape[0] == len(tuning_train),
        "tuning-training arrays are not row-aligned",
    )
    require(
        x_tuning_valid.shape[0] == y_tuning_valid.shape[0] == len(tuning_valid),
        "tuning-validation arrays are not row-aligned",
    )
    require(
        y_tuning_train.shape[1] == y_tuning_valid.shape[1] == len(cache.genes),
        "tuning target gene order/count mismatch",
    )

    search_rows: list[dict] = []
    for min_samples_leaf in leaf_grid:
        name = config_name(min_samples_leaf)
        config_root = tuning_root / name
        config_root.mkdir(parents=True, exist_ok=True)
        metrics_path = config_root / "tuning_validation_metrics.json"
        completed_path = config_root / "result.json"
        parameters = model_parameters(
            rf_contract, min_samples_leaf, bool(args.smoke_test)
        )
        if completed_path.is_file():
            completed = json.loads(completed_path.read_text())
            require(completed["status"] == "CONFIG_OK", f"incomplete RF tuning config: {name}")
            require(completed["run_fingerprint"] == run_fingerprint, f"RF tuning provenance changed: {name}")
            require(completed["model_parameters"] == parameters, f"RF tuning parameters changed: {name}")
            require(completed["tuning_model_retained"] is False, f"RF tuning model retained: {name}")
            require(metrics_path.is_file(), f"RF tuning metrics missing: {name}")
            require(
                completed["selection_row"]["metrics_sha256"] == sha256(metrics_path),
                f"RF tuning metrics hash changed: {name}",
            )
            search_rows.append(completed["selection_row"])
            continue

        model = make_forest(parameters)
        resources_before = resource_snapshot()
        started_utc = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        model.fit(x_tuning_train, y_tuning_train)
        fit_seconds = time.monotonic() - started
        prediction = np.asarray(model.predict(x_tuning_valid), dtype=np.float32)
        require(prediction.shape == y_tuning_valid.shape, f"RF tuning prediction shape mismatch: {name}")
        require(np.isfinite(prediction).all(), f"RF tuning prediction is non-finite: {name}")
        metrics = regression_metrics(y_tuning_valid, prediction, tuning_valid_controls)
        require(np.isfinite(metrics["delta_mse"]), f"RF tuning delta MSE is non-finite: {name}")
        total_seconds = time.monotonic() - started
        del model, prediction
        gc.collect()
        metrics_payload = {
            "status": "TUNING_VALIDATION_OK",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "selection_metric": "tuning-validation delta-expression MSE",
            "metrics": metrics,
            "model_parameters": parameters,
            "tuning_train_conditions": int(len(tuning_train)),
            "tuning_validation_conditions": int(len(tuning_valid)),
            "formal_validation_used": False,
            "tuning_model_retained": False,
            "test_response_accessed": False,
            "smoke_test": bool(args.smoke_test),
        }
        write_json_atomic(metrics_path, metrics_payload)
        selection_row = {
            "config": name,
            "min_samples_leaf": min_samples_leaf,
            "n_estimators": parameters["n_estimators"],
            "max_depth": parameters["max_depth"],
            "tuning_validation_delta_mse": metrics["delta_mse"],
            "tuning_validation_median_condition_delta_pearson": metrics[
                "median_condition_delta_pearson"
            ],
            "tuning_validation_mean_condition_absolute_r2": metrics[
                "mean_condition_absolute_r2"
            ],
            "metrics": str(metrics_path),
            "metrics_sha256": sha256(metrics_path),
            "fit_seconds": float(fit_seconds),
            "fit_and_validate_seconds": float(total_seconds),
        }
        write_json_atomic(
            completed_path,
            {
                "status": "CONFIG_OK",
                "started_utc": started_utc,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "run_fingerprint": run_fingerprint,
                "model_parameters": parameters,
                "selection_row": selection_row,
                "tuning_model_retained": False,
                "resource_before": resources_before,
                "resource_after": resource_snapshot(),
                "smoke_test": bool(args.smoke_test),
            },
        )
        search_rows.append(selection_row)

    require(
        not list(tuning_root.rglob("*.joblib")),
        "tuning model artifacts must not be retained",
    )
    search = (
        pd.DataFrame(search_rows)
        .sort_values(["tuning_validation_delta_mse", "config"], kind="stable")
        .reset_index(drop=True)
    )
    require(len(search) == 3 and search["config"].is_unique, "RF grid is incomplete")
    summary_columns = [
        "config",
        "min_samples_leaf",
        "n_estimators",
        "max_depth",
        "tuning_validation_delta_mse",
        "tuning_validation_median_condition_delta_pearson",
        "tuning_validation_mean_condition_absolute_r2",
        "metrics",
        "metrics_sha256",
        "fit_seconds",
        "fit_and_validate_seconds",
    ]
    require(
        set(search.columns) == set(summary_columns),
        "RF tuning-summary columns differ from the output contract",
    )
    search = search.loc[:, summary_columns]
    summary_path = tuning_root / "search_summary.csv"
    write_csv_atomic(summary_path, search)
    best = search.iloc[0].to_dict()

    hyperparameter_path = result_root / "frozen_hyperparameters.json"
    proposed_hyperparameters = {
        "status": "HYPERPARAMETERS_FROZEN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "tuning-validation delta-expression MSE",
        "selection_mode": "min",
        "selected_config": best["config"],
        "selected_min_samples_leaf": int(best["min_samples_leaf"]),
        "selected_tuning_validation_delta_mse": float(
            best["tuning_validation_delta_mse"]
        ),
        "grid": leaf_grid,
        "configuration_count": 3,
        "development_split_sha256": sha256(development_used_path),
        "search_summary": str(summary_path),
        "search_summary_sha256": sha256(summary_path),
        "formal_validation_used_for_selection": False,
        "tuning_models_retained": 0,
        "run_fingerprint": run_fingerprint,
        "smoke_test": bool(args.smoke_test),
        "test_response_accessed": False,
    }
    if hyperparameter_path.is_file():
        frozen_hyperparameters = json.loads(hyperparameter_path.read_text())
        require(
            frozen_hyperparameters["status"] == "HYPERPARAMETERS_FROZEN"
            and frozen_hyperparameters["selected_config"] == best["config"]
            and int(frozen_hyperparameters["selected_min_samples_leaf"])
            == int(best["min_samples_leaf"])
            and np.isclose(
                float(frozen_hyperparameters["selected_tuning_validation_delta_mse"]),
                float(best["tuning_validation_delta_mse"]),
                rtol=1e-12,
                atol=1e-15,
            )
            and frozen_hyperparameters["search_summary_sha256"] == sha256(summary_path)
            and frozen_hyperparameters["run_fingerprint"] == run_fingerprint,
            "recomputed RF hyperparameter selection differs from the frozen record",
        )
    else:
        write_json_atomic(hyperparameter_path, proposed_hyperparameters)
        frozen_hyperparameters = proposed_hyperparameters

    del x_tuning_train, y_tuning_train, x_tuning_valid, y_tuning_valid
    del tuning_valid_controls
    gc.collect()

    selected_leaf = int(frozen_hyperparameters["selected_min_samples_leaf"])
    final_parameters = model_parameters(rf_contract, selected_leaf, bool(args.smoke_test))
    final_model_path = model_root / "final_random_forest.joblib"
    artifact_path = model_root / "final_random_forest_artifact.json"
    formal_metrics_path = result_root / "formal_validation_metrics.json"
    final_fit_path = result_root / "final_fit.json"

    if final_fit_path.is_file():
        final_fit = json.loads(final_fit_path.read_text())
        require(final_fit["status"] == "FINAL_FIT_OK", "prior RF final fit is incomplete")
        require(final_fit["run_fingerprint"] == run_fingerprint, "RF final-fit provenance changed")
        require(final_fit["model_parameters"] == final_parameters, "RF final-fit parameters changed")
        require(final_model_path.is_file(), "RF final model is missing")
        require(final_fit["model_sha256"] == sha256(final_model_path), "RF final model hash changed")
        require(formal_metrics_path.is_file(), "RF formal-validation metrics are missing")
        require(
            final_fit["formal_validation_metrics_sha256"] == sha256(formal_metrics_path),
            "RF formal-validation metric hash changed",
        )
        formal_metrics_payload = json.loads(formal_metrics_path.read_text())
    else:
        x_formal_valid = np.asarray(design[formal_valid], dtype=np.float32)
        y_formal_valid = np.asarray(delta[formal_valid], dtype=np.float32)
        formal_valid_controls = np.asarray(
            cache.controls[cache.control_index[formal_valid]], dtype=np.float32
        )
        if artifact_path.is_file():
            artifact = json.loads(artifact_path.read_text())
            require(artifact["status"] == "FINAL_MODEL_SAVED", "RF final-model artifact is incomplete")
            require(artifact["run_fingerprint"] == run_fingerprint, "RF final-model provenance changed")
            require(artifact["model_parameters"] == final_parameters, "RF final-model parameters changed")
            require(final_model_path.is_file(), "RF final-model artifact points to a missing model")
            require(artifact["model_sha256"] == sha256(final_model_path), "RF final-model hash changed")
            fit_seconds = float(artifact["fit_seconds"])
            fit_started_utc = artifact["fit_started_utc"]
        else:
            x_formal_train = np.asarray(design[formal_train], dtype=np.float32)
            y_formal_train = np.asarray(delta[formal_train], dtype=np.float32)
            require(
                x_formal_train.shape[0] == y_formal_train.shape[0] == len(formal_train),
                "formal-training arrays are not row-aligned",
            )
            final_model = make_forest(final_parameters)
            fit_started_utc = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            final_model.fit(x_formal_train, y_formal_train)
            fit_seconds = time.monotonic() - started
            dump_joblib_atomic(final_model_path, final_model)
            del final_model, x_formal_train, y_formal_train
            gc.collect()
            artifact = {
                "status": "FINAL_MODEL_SAVED",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "fit_started_utc": fit_started_utc,
                "fit_seconds": float(fit_seconds),
                "fit_scope": "all formal internal-training conditions",
                "fit_conditions": int(len(formal_train)),
                "model_parameters": final_parameters,
                "model": str(final_model_path),
                "model_sha256": sha256(final_model_path),
                "run_fingerprint": run_fingerprint,
                "test_response_accessed": False,
                "smoke_test": bool(args.smoke_test),
            }
            write_json_atomic(artifact_path, artifact)

        reloaded_model = joblib.load(final_model_path)
        prediction = np.asarray(reloaded_model.predict(x_formal_valid), dtype=np.float32)
        require(prediction.shape == y_formal_valid.shape, "RF formal-validation prediction shape mismatch")
        require(np.isfinite(prediction).all(), "RF formal-validation prediction is non-finite")
        formal_metrics = regression_metrics(
            y_formal_valid, prediction, formal_valid_controls
        )
        formal_metrics_payload = {
            "status": "FORMAL_VALIDATION_DIAGNOSTIC_OK",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "role": "post-selection diagnostic only",
            "used_for_hyperparameter_selection": False,
            "metrics": formal_metrics,
            "model_parameters": final_parameters,
            "model": str(final_model_path),
            "model_sha256": sha256(final_model_path),
            "formal_train_conditions": int(len(formal_train)),
            "formal_validation_conditions": int(len(formal_valid)),
            "validation_source": "atomically saved and reloaded final joblib model",
            "test_response_accessed": False,
            "smoke_test": bool(args.smoke_test),
        }
        write_json_atomic(formal_metrics_path, formal_metrics_payload)
        final_fit = {
            "status": "FINAL_FIT_OK",
            "fit_started_utc": fit_started_utc,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "fit_seconds": float(fit_seconds),
            "fit_scope": "all formal internal-training conditions",
            "formal_train_conditions": int(len(formal_train)),
            "formal_validation_conditions": int(len(formal_valid)),
            "formal_validation_role": "post-selection diagnostic only",
            "model_parameters": final_parameters,
            "model": str(final_model_path),
            "model_sha256": sha256(final_model_path),
            "formal_validation_metrics": str(formal_metrics_path),
            "formal_validation_metrics_sha256": sha256(formal_metrics_path),
            "run_fingerprint": run_fingerprint,
            "test_response_accessed": False,
            "smoke_test": bool(args.smoke_test),
            "resource_final": resource_snapshot(),
        }
        write_json_atomic(final_fit_path, final_fit)
        del reloaded_model, prediction, x_formal_valid, y_formal_valid
        del formal_valid_controls
        gc.collect()

    require(final_model_path.is_file(), "RF final model does not exist")
    require(formal_metrics_path.is_file(), "RF formal-validation metrics do not exist")
    require(final_fit_path.is_file(), "RF final-fit completion record does not exist")
    formal_metrics_payload = json.loads(formal_metrics_path.read_text())
    require(
        formal_metrics_payload["status"] == "FORMAL_VALIDATION_DIAGNOSTIC_OK",
        "RF formal-validation diagnostics did not pass",
    )

    selection_path = result_root / "selection.json"
    proposed_selection = {
        "status": "SELECTION_FROZEN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "two-stage development selection followed by one all-training final fit",
        "selection_metric": "tuning-validation delta-expression MSE",
        "selection_mode": "min",
        "selected_config": frozen_hyperparameters["selected_config"],
        "selected_min_samples_leaf": selected_leaf,
        "selected_tuning_validation_delta_mse": frozen_hyperparameters[
            "selected_tuning_validation_delta_mse"
        ],
        "final_model": str(final_model_path),
        "final_model_sha256": sha256(final_model_path),
        "formal_validation_metrics": str(formal_metrics_path),
        "formal_validation_metrics_sha256": sha256(formal_metrics_path),
        "formal_validation_role": "post-selection diagnostic only",
        "formal_validation_delta_mse": formal_metrics_payload["metrics"]["delta_mse"],
        "formal_train_conditions": int(len(formal_train)),
        "formal_validation_conditions": int(len(formal_valid)),
        "tuning_train_conditions": int(len(tuning_train)),
        "tuning_validation_conditions": int(len(tuning_valid)),
        "tuning_models_retained": 0,
        "frozen_hyperparameters": str(hyperparameter_path),
        "frozen_hyperparameters_sha256": sha256(hyperparameter_path),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "development_split_sha256": sha256(development_used_path),
        "run_fingerprint": run_fingerprint,
        "test_response_accessed": False,
        "smoke_test": bool(args.smoke_test),
    }
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        require(selection["status"] == "SELECTION_FROZEN", "prior RF selection is not frozen")
        require(
            selection["run_fingerprint"] == run_fingerprint
            and selection["selected_min_samples_leaf"] == selected_leaf
            and selection["final_model_sha256"] == sha256(final_model_path)
            and selection["formal_validation_metrics_sha256"]
            == sha256(formal_metrics_path),
            "recomputed RF final selection differs from the frozen selection",
        )
    else:
        write_json_atomic(selection_path, proposed_selection)
        selection = proposed_selection

    final_checks = {
        "status": "RF_TWO_STAGE_SEARCH_OK",
        "grid_configuration_count": int(len(search)),
        "grid_configuration_names": sorted(search["config"].tolist()),
        "grid_selection_is_argmin": bool(
            selection["selected_config"] == search.iloc[0]["config"]
            and int(selection["selected_min_samples_leaf"])
            == int(search.iloc[0]["min_samples_leaf"])
        ),
        "tuning_models_retained": int(len(list(tuning_root.rglob("*.joblib")))),
        "final_model_count": int(final_model_path.is_file()),
        "final_model_reloaded_for_diagnostics": True,
        "formal_validation_used_for_grid_selection": False,
        "formal_validation_metrics_present_before_selection_freeze": True,
        "selection_status": selection["status"],
        "test_response_accessed": False,
        "smoke_test": bool(args.smoke_test),
    }
    require(final_checks["grid_selection_is_argmin"], "RF selected config is not tuning-MSE argmin")
    require(final_checks["tuning_models_retained"] == 0, "RF tuning models were retained")
    require(final_checks["final_model_count"] == 1, "RF final-model count is not one")
    write_json_atomic(check_root / "final_checks.json", final_checks)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
