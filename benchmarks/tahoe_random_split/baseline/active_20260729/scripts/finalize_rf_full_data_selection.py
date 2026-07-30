#!/usr/bin/env python3
"""Audit and freeze the versioned full-data RF selection.

This program does not train or evaluate a model and never opens the external
OOD/test cache.  It may run only after run_rf_full_data_validation_search.py
has completed all three min_samples_leaf candidates.  The earlier
results/rf/selection.json is read-only upstream provenance and is never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


RUN = Path(__file__).resolve().parents[1]
LEAF_SIZES = (5, 20, 50)
FORMAL_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 10,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": 24,
    "random_state": 42,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_strings(values: list[str] | pd.Series | np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict:
    require(path.is_file(), f"{label} is missing: {path}")
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": sha256(path),
    }


def expected_model_path(run_root: Path, search_root: Path, leaf: int) -> Path:
    if leaf == 5:
        return (run_root / "models/rf/final_random_forest.joblib").resolve()
    return (
        search_root / f"min_samples_leaf_{leaf}/random_forest.joblib"
    ).resolve()


def expected_parameters(base: dict, leaf: int) -> dict:
    return {**base, "min_samples_leaf": int(leaf)}


def verify_parameter_record(
    observed: dict, expected: dict, label: str
) -> None:
    require(observed == expected, f"{label} model parameters changed")


def scalar_close(
    observed: object,
    expected: object,
    label: str,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> None:
    require(
        np.isclose(float(observed), float(expected), rtol=rtol, atol=atol),
        f"{label} differs: {observed} versus {expected}",
    )


def verify_comparison_row(
    row: pd.Series,
    *,
    expected: dict,
    model_path: Path,
    model_hash: str,
    metrics: dict,
    label: str,
) -> None:
    require(int(row["min_samples_leaf"]) == expected["min_samples_leaf"], f"{label} leaf differs")
    for key in (
        "n_estimators",
        "max_depth",
        "n_jobs",
        "random_state",
    ):
        require(int(row[key]) == int(expected[key]), f"{label} {key} differs")
    require(str(row["max_features"]) == str(expected["max_features"]), f"{label} max_features differs")
    require(
        str(row["bootstrap"]).strip().lower()
        == ("true" if expected["bootstrap"] else "false"),
        f"{label} bootstrap differs",
    )
    require(Path(str(row["model"])).resolve() == model_path, f"{label} model path differs")
    require(str(row["model_sha256"]) == model_hash, f"{label} model hash differs")
    metric_mapping = {
        "formal_validation_delta_mse": "delta_mse",
        "formal_validation_median_condition_delta_pearson": "median_condition_delta_pearson",
        "formal_validation_mean_condition_absolute_r2": "mean_condition_absolute_r2",
        "formal_validation_mean_condition_absolute_pearson": "mean_condition_absolute_pearson",
    }
    for column, key in metric_mapping.items():
        scalar_close(row[column], metrics[key], f"{label} {column}")


def verify_reloaded_model(
    path: Path,
    model_hash: str,
    parameters: dict,
    input_feature_count: int,
    output_count: int,
) -> dict:
    require(path.is_file(), f"RF model is missing: {path}")
    require(sha256(path) == model_hash, f"RF model hash changed: {path}")
    model = joblib.load(path)
    observed = model.get_params()
    for key, value in parameters.items():
        require(observed[key] == value, f"reloaded RF {key} differs for leaf={parameters['min_samples_leaf']}")
    require(int(model.n_features_in_) == input_feature_count, "reloaded RF input dimension differs")
    require(int(model.n_outputs_) == output_count, "reloaded RF output dimension differs")
    require(len(model.estimators_) == int(parameters["n_estimators"]), "reloaded RF fitted-tree count differs")
    return {
        "status": "MODEL_RELOAD_OK",
        "model": str(path),
        "model_sha256": model_hash,
        "input_feature_count": int(model.n_features_in_),
        "output_count": int(model.n_outputs_),
        "fitted_tree_count": int(len(model.estimators_)),
        "parameters": parameters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=RUN / "sensitivity/rf_full_data_validation_search_20260727",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=RUN / "cache/train_control_search"
    )
    parser.add_argument(
        "--contract", type=Path, default=RUN / "config/search_contract.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN / "results/rf/full_data_validated_selection_20260727.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=RUN / "provenance/rf/full_data_model_selection_audit_20260727.md",
    )
    parser.add_argument(
        "--synthetic-test-mode",
        action="store_true",
        help="Relax formal dimensions only for a fully /tmp-scoped test tree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    search_root = args.search_root.resolve()
    cache_root = args.cache_root.resolve()
    contract_path = args.contract.resolve()
    output_path = args.output.resolve()
    audit_path = args.audit.resolve()

    if args.synthetic_test_mode:
        temporary = Path("/tmp").resolve()
        for path in (
            run_root,
            search_root,
            cache_root,
            contract_path,
            output_path,
            audit_path,
        ):
            require(
                path.is_relative_to(temporary),
                "synthetic finalizer mode is restricted to /tmp",
            )
    else:
        require(run_root == RUN.resolve(), "formal run root differs from the RF baseline root")
        require(
            search_root
            == (RUN / "sensitivity/rf_full_data_validation_search_20260727").resolve(),
            "formal search root differs from the versioned RF search root",
        )
        require(
            cache_root == (RUN / "cache/train_control_search").resolve(),
            "formal cache root differs from the frozen training/control cache",
        )
        require(
            contract_path == (RUN / "config/search_contract.json").resolve(),
            "formal RF search-contract path changed",
        )
        require(
            output_path
            == (RUN / "results/rf/full_data_validated_selection_20260727.json").resolve(),
            "formal authoritative-selection path changed",
        )
        require(
            audit_path
            == (RUN / "provenance/rf/full_data_model_selection_audit_20260727.md").resolve(),
            "formal audit path changed",
        )

    # Never overwrite an earlier authoritative publication.
    if output_path.exists():
        require(audit_path.is_file(), "authoritative RF selection exists without its audit")
        existing = read_json(output_path, "existing authoritative RF selection")
        require(
            existing.get("status") == "FULL_DATA_RF_SELECTION_FROZEN",
            "existing authoritative RF selection has an unexpected status",
        )
        require(
            existing.get("test_response_accessed") is False
            and existing.get("ood_results_used_for_selection") is False,
            "existing authoritative RF selection does not preserve the OOD gate",
        )
        existing_audit = existing["audit"]
        require(
            Path(str(existing_audit["path"])).resolve() == audit_path
            and existing_audit["sha256"] == sha256(audit_path),
            "existing authoritative RF audit path or hash differs",
        )
        existing_model = Path(str(existing["selected"]["model"])).resolve()
        require(
            existing_model.is_file()
            and existing["selected"]["model_sha256"] == sha256(existing_model),
            "existing authoritative RF selected-model hash differs",
        )
        existing_upstream = existing["upstream_selection"]
        existing_upstream_path = Path(str(existing_upstream["path"])).resolve()
        require(
            existing_upstream_path.is_file()
            and existing_upstream["sha256"] == sha256(existing_upstream_path),
            "existing authoritative RF upstream-selection hash differs",
        )
        print(
            json.dumps(
                {
                    "status": "ALREADY_FROZEN_NO_WRITE",
                    "selection": str(output_path),
                    "selection_sha256": sha256(output_path),
                    "audit": str(audit_path),
                    "audit_sha256": sha256(audit_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    contract = read_json(contract_path, "RF search contract")
    rf_contract = contract["random_forest"]
    base_parameters = {
        key: rf_contract[key]
        for key in (
            "n_estimators",
            "max_depth",
            "max_features",
            "bootstrap",
            "n_jobs",
            "random_state",
        )
    }
    base_parameters["n_estimators"] = int(base_parameters["n_estimators"])
    base_parameters["n_jobs"] = int(base_parameters["n_jobs"])
    base_parameters["random_state"] = int(base_parameters["random_state"])
    if not args.synthetic_test_mode:
        require(base_parameters == FORMAL_PARAMETERS, "formal RF fixed parameters changed")
        require(
            [int(value) for value in rf_contract["min_samples_leaf"]]
            == list(LEAF_SIZES),
            "formal RF leaf grid changed",
        )

    cache_manifest_path = cache_root / "cache_manifest.json"
    cache_manifest = read_json(cache_manifest_path, "training/control cache manifest")
    require(
        cache_manifest.get("status") == "CACHE_OK"
        and cache_manifest.get("test_response_accessed") is False,
        "RF selection cache is not a test-free CACHE_OK artifact",
    )
    metadata_path = cache_root / "train/metadata.csv"
    genes_path = cache_root / "genes.csv"
    design_path = cache_root / "model_inputs/rf_design.npy"
    delta_path = cache_root / "model_inputs/delta_expression.npy"
    for path in (metadata_path, genes_path, design_path, delta_path):
        require(path.is_file(), f"training RF input is missing: {path}")
    metadata = pd.read_csv(metadata_path)
    genes = pd.read_csv(genes_path)["gene"].astype(str)
    require(metadata["condition_id"].is_unique, "training condition IDs are duplicated")
    require(genes.is_unique, "RF gene names are duplicated")
    train = metadata["search_split"].astype(str).eq("train")
    validation = metadata["search_split"].astype(str).eq("valid")
    require(not np.any(train & validation), "RF training/validation masks overlap")
    expected_train = int(train.sum())
    expected_validation = int(validation.sum())
    if not args.synthetic_test_mode:
        require(expected_train == 29_277, "formal RF train count differs from 29,277")
        require(expected_validation == 3_252, "formal RF validation count differs from 3,252")
        require(len(genes) == 13_784, "formal RF gene count differs from 13,784")
    train_ids_hash = hash_strings(metadata.loc[train, "condition_id"].astype(str))
    validation_ids_hash = hash_strings(
        metadata.loc[validation, "condition_id"].astype(str)
    )
    design = np.load(design_path, mmap_mode="r")
    delta = np.load(delta_path, mmap_mode="r")
    require(design.shape[0] == len(metadata), "RF design row count differs from metadata")
    require(delta.shape == (len(metadata), len(genes)), "RF delta shape differs from metadata/genes")
    expected_input_count = int(design.shape[1])
    if not args.synthetic_test_mode:
        require(expected_input_count == 1_927, "formal RF input dimension differs from 1,927")

    analysis_contract_path = search_root / "provenance/analysis_contract.json"
    analysis = read_json(analysis_contract_path, "RF full-data analysis contract")
    require(
        analysis.get("status") == "RF_FULL_DATA_VALIDATION_CONTRACT",
        "RF full-data analysis contract did not pass",
    )
    require(
        analysis.get("unit_of_analysis") == contract.get("unit_of_analysis")
        and analysis.get("target") == contract.get("target"),
        "RF unit of analysis or response target differs between contracts",
    )
    require(
        analysis.get("test_response_accessed") is False,
        "RF full-data analysis contract reports test access",
    )
    require(bool(analysis.get("smoke_test")) is args.synthetic_test_mode, "RF analysis mode differs")
    require(
        int(analysis["formal_train_conditions"]) == expected_train
        and int(analysis["formal_validation_conditions"]) == expected_validation,
        "RF analysis train/validation counts differ",
    )
    require(analysis["training_validation_overlap"] == 0, "RF analysis reports split overlap")
    require(
        analysis["formal_train_ids_sha256"] == train_ids_hash,
        "RF formal-training membership hash differs",
    )
    require(
        analysis["formal_validation_ids_sha256"] == validation_ids_hash,
        "RF formal-validation membership hash differs",
    )
    require(
        analysis["selection_metric"]
        == "formal-validation matched-control-relative delta-expression MSE",
        "RF selection metric changed",
    )
    require(
        analysis["cache_manifest_sha256"] == sha256(cache_manifest_path),
        "RF analysis cache-manifest hash differs",
    )
    require(
        analysis["training_metadata_sha256"] == sha256(metadata_path),
        "RF analysis metadata hash differs",
    )
    require(analysis["genes_sha256"] == sha256(genes_path), "RF analysis gene hash differs")
    require(
        analysis["delta_expression_sha256"] == sha256(delta_path),
        "RF analysis delta-expression hash differs",
    )
    require(
        analysis["rf_design_sha256"] == sha256(design_path),
        "RF analysis design hash differs",
    )
    search_script_path = run_root / "scripts/run_rf_full_data_validation_search.py"
    require(
        analysis["script_sha256"] == sha256(search_script_path),
        "RF full-data search script changed after contract publication",
    )
    if not args.synthetic_test_mode:
        require(
            analysis["min_samples_leaf_compared"] == list(LEAF_SIZES),
            "RF analysis leaf candidates changed",
        )
        require(
            analysis["new_models_fitted"] == [20, 50]
            and analysis["formal_leaf5_reused_read_only"] is True,
            "RF leaf reuse/new-fit contract changed",
        )
        require(
            analysis["fixed_parameters"] == FORMAL_PARAMETERS,
            "RF analysis fixed parameters changed",
        )
        feature_audit = analysis["feature_audit"]
        require(feature_audit["status"] == "FORMAL_FEATURES_MATCH", "RF feature audit did not pass")
        require(
            int(feature_audit["condition_feature_count"]) == 427
            and int(feature_audit["selected_control_feature_count"]) == 1_500
            and int(feature_audit["rf_input_feature_count"]) == expected_input_count,
            "RF feature dimensions changed",
        )
        encoder_path = run_root / "provenance/rf/condition_encoder.json"
        selected_features_path = (
            run_root / "provenance/rf/selected_train_only_control_features.csv"
        )
        require(
            feature_audit["condition_encoder_sha256"] == sha256(encoder_path),
            "RF condition-encoder hash differs",
        )
        require(
            feature_audit["selected_features_sha256"]
            == sha256(selected_features_path),
            "RF selected-control-feature hash differs",
        )
        require(
            feature_audit["rf_design_sha256"] == sha256(design_path),
            "RF feature-audit design hash differs",
        )
        features = pd.read_csv(selected_features_path)
        require(len(features) == 1_500, "RF selected-control feature count differs")
        indices = features["gene_index_zero_based"].to_numpy(dtype=np.int64)
        require(
            len(np.unique(indices)) == 1_500
            and np.all(indices >= 0)
            and np.all(indices < len(genes)),
            "RF selected-control gene indices are invalid",
        )
        require(
            genes.iloc[indices].tolist() == features["gene"].astype(str).tolist(),
            "RF selected-control genes differ from gene order",
        )

    analysis_hash = sha256(analysis_contract_path)
    comparison_path = search_root / "full_data_validation_comparison.csv"
    summary_path = search_root / "full_data_validation_comparison.json"
    require(comparison_path.is_file(), "RF full-data comparison is incomplete")
    require(summary_path.is_file(), "RF full-data comparison summary is incomplete")
    comparison = pd.read_csv(comparison_path)
    require(len(comparison) == 3, "RF full-data comparison must contain exactly three rows")
    require(
        set(comparison["min_samples_leaf"].astype(int)) == set(LEAF_SIZES),
        "RF full-data comparison leaf set differs",
    )
    require(
        comparison["min_samples_leaf"].astype(int).is_unique,
        "RF full-data comparison duplicates a leaf",
    )
    mse = comparison["formal_validation_delta_mse"].to_numpy(dtype=np.float64)
    require(np.isfinite(mse).all(), "RF comparison contains non-finite validation MSE")
    minimum = float(np.min(mse))
    require(int(np.sum(mse == minimum)) == 1, "RF validation MSE argmin is tied")
    best = comparison.iloc[int(np.argmin(mse))]

    leaf5_audit_path = search_root / "provenance/leaf5_reuse_audit.json"
    leaf5_audit = read_json(leaf5_audit_path, "leaf=5 reuse audit")
    require(
        leaf5_audit.get("status") == "LEAF5_REUSE_AUDIT_OK"
        and leaf5_audit.get("test_response_accessed") is False,
        "leaf=5 reuse audit did not pass without test access",
    )
    require(
        int(leaf5_audit["formal_train_conditions"]) == expected_train
        and int(leaf5_audit["formal_validation_conditions"]) == expected_validation,
        "leaf=5 reuse counts differ",
    )
    require(
        leaf5_audit["formal_train_ids_sha256"] == train_ids_hash
        and leaf5_audit["formal_validation_ids_sha256"] == validation_ids_hash,
        "leaf=5 reuse split membership differs",
    )

    model_audits: dict[str, dict] = {}
    upstream_selection_path = run_root / "results/rf/selection.json"
    upstream_selection = read_json(upstream_selection_path, "upstream RF selection")
    require(
        upstream_selection.get("status") == "SELECTION_FROZEN"
        and upstream_selection.get("test_response_accessed") is False,
        "upstream RF selection is not a frozen test-free record",
    )

    for leaf in LEAF_SIZES:
        label = f"leaf={leaf}"
        row = comparison.loc[comparison["min_samples_leaf"].astype(int).eq(leaf)]
        require(len(row) == 1, f"{label} comparison lookup is not one-to-one")
        row = row.iloc[0]
        parameters = expected_parameters(base_parameters, leaf)
        model_path = expected_model_path(run_root, search_root, leaf)
        model_hash = sha256(model_path)

        if leaf == 5:
            require(
                leaf5_audit["model_sha256"] == model_hash
                and Path(str(leaf5_audit["model"])).resolve() == model_path,
                "leaf=5 reuse model path or hash differs",
            )
            verify_parameter_record(
                leaf5_audit["model_parameters"], parameters, "leaf=5 reuse"
            )
            formal_metrics_path = (
                run_root / "results/rf/formal_validation_metrics.json"
            )
            formal_metrics = read_json(
                formal_metrics_path, "leaf=5 formal-validation metrics"
            )
            require(
                formal_metrics.get("status")
                == "FORMAL_VALIDATION_DIAGNOSTIC_OK"
                and formal_metrics.get("test_response_accessed") is False,
                "leaf=5 formal validation did not pass",
            )
            require(
                int(formal_metrics["formal_train_conditions"]) == expected_train
                and int(formal_metrics["formal_validation_conditions"])
                == expected_validation,
                "leaf=5 formal metric counts differ",
            )
            require(
                formal_metrics["model_sha256"] == model_hash
                and Path(str(formal_metrics["model"])).resolve() == model_path,
                "leaf=5 formal metric model path or hash differs",
            )
            verify_parameter_record(
                formal_metrics["model_parameters"], parameters, "leaf=5 formal metrics"
            )
            metrics = formal_metrics["metrics"]
            scalar_close(
                metrics["delta_mse"],
                leaf5_audit["recomputed_validation_delta_mse"],
                "leaf=5 recomputed validation MSE",
                rtol=1e-10,
                atol=1e-12,
            )
            require(
                upstream_selection["final_model_sha256"] == model_hash
                and Path(str(upstream_selection["final_model"])).resolve()
                == model_path,
                "upstream RF selection no longer identifies the leaf=5 model",
            )
        else:
            candidate_root = search_root / f"min_samples_leaf_{leaf}"
            artifact_path = candidate_root / "model_artifact.json"
            metrics_path = candidate_root / "formal_validation_metrics.json"
            result_path = candidate_root / "result.json"
            artifact = read_json(artifact_path, f"{label} model artifact")
            metric_payload = read_json(metrics_path, f"{label} validation metrics")
            result = read_json(result_path, f"{label} result")
            require(
                artifact.get("status") == "MODEL_SAVED"
                and artifact.get("test_response_accessed") is False,
                f"{label} model artifact did not pass",
            )
            require(
                metric_payload.get("status") == "FORMAL_VALIDATION_OK"
                and metric_payload.get("test_response_accessed") is False,
                f"{label} validation metrics did not pass",
            )
            require(
                result.get("status") == "CANDIDATE_OK"
                and result.get("test_response_accessed") is False,
                f"{label} result did not pass",
            )
            for payload_name, payload in (
                ("artifact", artifact),
                ("metrics", metric_payload),
                ("result", result),
            ):
                require(
                    payload["provenance_sha256"] == analysis_hash,
                    f"{label} {payload_name} provenance hash differs",
                )
                require(
                    payload["model_sha256"] == model_hash,
                    f"{label} {payload_name} model hash differs",
                )
                verify_parameter_record(
                    payload["model_parameters"],
                    parameters,
                    f"{label} {payload_name}",
                )
                require(
                    Path(str(payload["model"])).resolve() == model_path,
                    f"{label} {payload_name} model path differs",
                )
            require(
                int(artifact["fit_conditions"]) == expected_train,
                f"{label} fit count differs",
            )
            require(
                int(metric_payload["training_conditions"]) == expected_train
                and int(metric_payload["validation_conditions"])
                == expected_validation,
                f"{label} metric counts differ",
            )
            require(
                result["metrics_sha256"] == sha256(metrics_path),
                f"{label} metric-file hash differs",
            )
            require(
                Path(str(result["model"])).resolve() == model_path,
                f"{label} result model path differs",
            )
            metrics = metric_payload["metrics"]
            result_row = result["comparison_row"]
            scalar_close(
                result_row["formal_validation_delta_mse"],
                metrics["delta_mse"],
                f"{label} result-row validation MSE",
            )

        verify_comparison_row(
            row,
            expected=parameters,
            model_path=model_path,
            model_hash=model_hash,
            metrics=metrics,
            label=label,
        )
        model_audits[str(leaf)] = verify_reloaded_model(
            model_path,
            model_hash,
            parameters,
            expected_input_count,
            len(genes),
        )

    summary = read_json(summary_path, "RF full-data comparison summary")
    require(
        summary.get("status") == "RF_FULL_DATA_VALIDATION_SEARCH_OK"
        and summary.get("test_response_accessed") is False,
        "RF full-data comparison summary did not pass",
    )
    require(
        summary.get("formal_selection_modified") is False,
        "RF full-data runner reports modifying the upstream selection",
    )
    require(
        summary["comparison_sha256"] == sha256(comparison_path),
        "RF comparison-summary table hash differs",
    )
    require(
        int(summary["strict_argmin_min_samples_leaf"])
        == int(best["min_samples_leaf"]),
        "RF comparison-summary argmin leaf differs",
    )
    scalar_close(
        summary["strict_argmin_validation_delta_mse"],
        best["formal_validation_delta_mse"],
        "RF comparison-summary argmin validation MSE",
    )
    selected_model_path = Path(str(best["model"])).resolve()
    require(
        Path(str(summary["strict_argmin_model"])).resolve()
        == selected_model_path,
        "RF comparison-summary selected-model path differs",
    )
    require(
        selected_model_path
        == expected_model_path(
            run_root, search_root, int(best["min_samples_leaf"])
        ),
        "RF strict-argmin model path differs",
    )
    require(
        summary["strict_argmin_model_sha256"] == sha256(selected_model_path),
        "RF comparison-summary selected-model hash differs",
    )

    upstream_selection_hash = sha256(upstream_selection_path)
    audit = f"""# Full-data RF model-selection audit

Status: **PASS**

- Analysis unit: drug-dose-cell-line condition.
- Target: treated-condition mean minus matched cell-line control mean.
- Formal internal training set: {expected_train:,} conditions; membership SHA256 `{train_ids_hash}`.
- Fixed validation set: {expected_validation:,} conditions; membership SHA256 `{validation_ids_hash}`.
- Input design: {expected_input_count:,} features; {len(genes):,} gene outputs.
- Fixed parameters: {base_parameters['n_estimators']} trees, maximum depth {base_parameters['max_depth']}, `sqrt` feature sampling, bootstrap enabled and random state {base_parameters['random_state']}.
- Compared `min_samples_leaf`: 5, 20 and 50.
- Selection metric: pooled formal-validation matched-control-relative delta-expression MSE.
- Strict argmin: `min_samples_leaf={int(best['min_samples_leaf'])}`.
- Selected validation delta-expression MSE: {float(best['formal_validation_delta_mse']):.12g}.
- Selected model SHA256: `{sha256(selected_model_path)}`.
- All three saved models were reloaded and their fitted dimensions, tree counts and fixed parameters were checked.
- External OOD responses were not used for this selection.

The earlier `results/rf/selection.json` remains unchanged and is retained as
upstream provenance. The versioned JSON published with this audit is
authoritative for the RF hyperparameter and model used in subsequent OOD
evaluation.
"""
    write_text_atomic(audit_path, audit)
    audit_hash = sha256(audit_path)
    record = {
        "status": "FULL_DATA_RF_SELECTION_FROZEN",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_for": (
            "RF min_samples_leaf and saved model used in subsequent OOD evaluation"
        ),
        "unit_of_analysis": contract["unit_of_analysis"],
        "target": contract["target"],
        "training_conditions": expected_train,
        "validation_conditions": expected_validation,
        "formal_train_ids_sha256": train_ids_hash,
        "formal_validation_ids_sha256": validation_ids_hash,
        "selection_metric": (
            "formal-validation delta-expression MSE pooled across condition-gene pairs"
        ),
        "selection_mode": "unique strict argmin",
        "candidate_min_samples_leaf": list(LEAF_SIZES),
        "fixed_parameters": base_parameters,
        "input_feature_count": expected_input_count,
        "output_gene_count": len(genes),
        "selected": {
            "min_samples_leaf": int(best["min_samples_leaf"]),
            "formal_validation_delta_mse": float(
                best["formal_validation_delta_mse"]
            ),
            "formal_validation_median_condition_delta_pearson": float(
                best["formal_validation_median_condition_delta_pearson"]
            ),
            "formal_validation_mean_condition_absolute_r2": float(
                best["formal_validation_mean_condition_absolute_r2"]
            ),
            "formal_validation_mean_condition_absolute_pearson": float(
                best["formal_validation_mean_condition_absolute_pearson"]
            ),
            "model": str(selected_model_path),
            "model_sha256": sha256(selected_model_path),
        },
        "search": {
            "analysis_contract": file_record(analysis_contract_path),
            "comparison": file_record(comparison_path),
            "comparison_summary": file_record(summary_path),
            "leaf5_reuse_audit": file_record(leaf5_audit_path),
        },
        "model_reload_audits": model_audits,
        "training_inputs": {
            "cache_manifest": file_record(cache_manifest_path),
            "metadata": file_record(metadata_path),
            "genes": file_record(genes_path),
            "rf_design": file_record(design_path),
            "delta_expression": file_record(delta_path),
        },
        "upstream_selection": {
            "path": str(upstream_selection_path),
            "sha256": upstream_selection_hash,
            "preserved_read_only": True,
        },
        "audit": {
            "path": str(audit_path),
            "sha256": audit_hash,
        },
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "test_response_accessed": False,
        "ood_results_used_for_selection": False,
    }
    require(
        sha256(upstream_selection_path) == upstream_selection_hash,
        "upstream RF selection changed during finalization",
    )
    write_json_atomic(output_path, record)
    require(read_json(output_path, "written RF authoritative selection") == record, "written RF selection changed")
    require(
        sha256(upstream_selection_path) == upstream_selection_hash,
        "upstream RF selection changed after finalization",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "selection": str(output_path),
                "selection_sha256": sha256(output_path),
                "audit": str(audit_path),
                "audit_sha256": audit_hash,
                "upstream_selection_modified": False,
                "test_response_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
