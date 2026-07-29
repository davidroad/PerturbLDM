#!/usr/bin/env python3
"""Update CPA rows in random performance tables with latest CPA results."""

from __future__ import annotations

import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("pipeline_results_gauss_random")
PREDICTION_INPUT = BASE / "random_merged_condition_metrics_exact_6methods_RUN.csv"
DISTRIBUTION_INPUT = BASE / "merged_metrics_distribution_random_diffusion_cpa_chemcpa.csv"
LATEST_PREDICTION_JSON = Path("random_inference_full_gauss/merged_metrics_by_condition.json")
LATEST_DISTRIBUTION_CSV = Path("random_distribution_similarity_gauss/global_condition_metrics.csv")

PREDICTION_OUTPUT = BASE / "random_merged_condition_metrics_exact_6methods_RUN_cpa_updated.csv"
DISTRIBUTION_OUTPUT = BASE / "merged_metrics_distribution_random_diffusion_cpa_chemcpa_cpa_updated.csv"
PREDICTION_SUMMARY_XLSX = BASE / "metrics_cpa_updated.xlsx"
DISTRIBUTION_SUMMARY_XLSX = BASE / "distribution_metrics_cpa_updated.xlsx"
CHANGE_SUMMARY_CSV = BASE / "cpa_random_performance_before_after_summary.csv"
PLOT_OUTPUT = BASE / "cpa_random_performance_before_after.png"


PREDICTION_METRICS = ["MSE", "MAE", "Pearson_r", "Spearman_r", "R2", "Chatterjee"]
DISTRIBUTION_METRICS = ["MMD_RBF", "E_distance", "Wasserstein_Sliced", "Wasserstein_OT"]
LOWER_IS_BETTER = {"MSE", "MAE", "MMD_RBF", "E_distance", "Wasserstein_Sliced", "Wasserstein_OT"}


def normalize_cell_line(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.startswith("CVCL_"):
        text = text.replace("_", "-", 1)
    return text


def normalize_dose(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    numeric_text = text.replace("-", ".")
    try:
        return f"{float(numeric_text):.12g}"
    except ValueError:
        return text


def normalize_drug(value: object) -> str:
    return "" if pd.isna(value) else re.sub(r"\s+", " ", str(value).strip())


def make_key(cell_line: object, drug: object, dose: object) -> str:
    return "\t".join([normalize_cell_line(cell_line), normalize_drug(drug), normalize_dose(dose)])


def parse_latest_prediction(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        data = json.load(handle)
    rows = []
    for condition, metrics in data["metrics_by_condition"].items():
        left, dose = condition.rsplit("_", 1)
        cell_line, drug = left.split("_", 1)
        rows.append(
            {
                "_match_key": make_key(cell_line, drug, dose),
                "MSE": metrics.get("mse"),
                "MAE": metrics.get("mae"),
                "R2": metrics.get("r2_score"),
                "Pearson_r": metrics.get("pearson_r"),
                "Spearman_r": metrics.get("spearman_r"),
                "Chatterjee": metrics.get("chatterjee_r"),
            }
        )
    latest = pd.DataFrame(rows).drop_duplicates("_match_key", keep="last")
    return latest.set_index("_match_key")


def parse_latest_distribution(path: Path) -> pd.DataFrame:
    latest = pd.read_csv(path, dtype=str, low_memory=False)
    for column in DISTRIBUTION_METRICS + ["n_real", "n_pred"]:
        latest[column] = pd.to_numeric(latest[column], errors="coerce")
    latest["_match_key"] = [
        make_key(cell_line, drug, dose)
        for cell_line, drug, dose in zip(latest["cell_line"], latest["drug"], latest["dose"])
    ]
    return latest.drop_duplicates("_match_key", keep="last").set_index("_match_key")


def summarize(df: pd.DataFrame, method_order: Iterable[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for method in method_order:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        row = {"method": method}
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows)


def cpa_summary_row(df: pd.DataFrame, method_value: str, metrics: list[str], group: str) -> list[dict]:
    sub = df[df["method"] == method_value]
    rows = []
    for metric in metrics:
        values = pd.to_numeric(sub[metric], errors="coerce")
        rows.append({"group": group, "metric": metric, "mean": values.mean(), "std": values.std()})
    return rows


def format_float(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def column_name(index: int) -> str:
    index += 1
    chars = []
    while index:
        index, remainder = divmod(index - 1, 26)
        chars.append(chr(65 + remainder))
    return "".join(reversed(chars))


def write_xlsx(path: Path, rows: list[list[object]], sheet_name: str = "Sheet1") -> None:
    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, raw_value in enumerate(row, start=1):
            cell_ref = f"{column_name(c_idx - 1)}{r_idx}"
            value = format_float(raw_value)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
                )
        sheet_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>"
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            "</Relationships>"
        ),
        "xl/worksheets/sheet1.xml": worksheet,
        "xl/styles.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Arial"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            "</styleSheet>"
        ),
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:creator>Codex</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
            "</cp:coreProperties>"
        ),
        "docProps/app.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>Codex</Application>"
            "</Properties>"
        ),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def summary_to_rows(summary: pd.DataFrame) -> list[list[object]]:
    return [summary.columns.tolist()] + summary.fillna("").values.tolist()


def plot_changes(change_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {"before": "#4C78A8", "after": "#F58518"}
    for row_idx, group in enumerate(["prediction", "distribution"]):
        sub = change_df[change_df["group"] == group].copy()
        labels = sub["metric"].tolist()
        x = range(len(labels))
        width = 0.36
        ax = axes[row_idx, 0]
        ax.bar([i - width / 2 for i in x], sub["before_mean"], width=width, label="before", color=colors["before"])
        ax.bar([i + width / 2 for i in x], sub["after_mean"], width=width, label="after", color=colors["after"])
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Mean value")
        ax.set_title(f"CPA {group} metrics")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        if group == "distribution":
            ax.set_yscale("log")

        ax2 = axes[row_idx, 1]
        bar_colors = ["#009E73" if v >= 0 else "#D55E00" for v in sub["improvement_percent"]]
        ax2.bar(labels, sub["improvement_percent"], color=bar_colors)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(list(range(len(labels))))
        ax2.set_xticklabels(labels, rotation=35, ha="right")
        ax2.set_ylabel("Improvement (%)")
        ax2.set_title("Positive means better")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.savefig(PLOT_OUTPUT, dpi=300)
    plt.close(fig)


def main() -> None:
    latest_prediction = parse_latest_prediction(LATEST_PREDICTION_JSON)
    latest_distribution = parse_latest_distribution(LATEST_DISTRIBUTION_CSV)

    prediction = pd.read_csv(
        PREDICTION_INPUT,
        dtype={"drug": str, "dose": str, "cellname": str, "method": str, "condition_final": str},
        low_memory=False,
    )
    old_prediction = prediction.copy()
    prediction_mask = prediction["method"].eq("CPA")
    prediction_keys = prediction.loc[prediction_mask].apply(
        lambda row: make_key(row["cellname"], row["drug"], row["dose"]), axis=1
    )
    prediction_matches = prediction_keys.isin(latest_prediction.index)
    prediction_matched_index = prediction_keys.index[prediction_matches]
    for column in PREDICTION_METRICS:
        mapped = prediction_keys.map(latest_prediction[column])
        prediction.loc[prediction_matched_index, column] = mapped.loc[prediction_matched_index].values

    distribution = pd.read_csv(
        DISTRIBUTION_INPUT,
        dtype={"cell_line": str, "cell_name": str, "dose": str, "drug": str, "CondID": str, "cond_id": str, "method": str},
        low_memory=False,
    )
    old_distribution = distribution.copy()
    distribution_mask = distribution["method"].str.lower().eq("cpa")
    distribution_keys = distribution.loc[distribution_mask].apply(
        lambda row: make_key(row["cell_line"], row["drug"], row["dose"]), axis=1
    )
    distribution_matches = distribution_keys.isin(latest_distribution.index)
    distribution_matched_index = distribution_keys.index[distribution_matches]
    for column in DISTRIBUTION_METRICS + ["n_real", "n_pred"]:
        mapped = distribution_keys.map(latest_distribution[column])
        distribution.loc[distribution_matched_index, column] = mapped.loc[distribution_matched_index].values

    prediction.to_csv(PREDICTION_OUTPUT, index=False)
    distribution.to_csv(DISTRIBUTION_OUTPUT, index=False)

    prediction_summary = summarize(
        prediction,
        ["Diffusion", "CPA", "MLP", "RF", "TrivalZero", "chemCPA", "scGen"],
        PREDICTION_METRICS,
    )
    distribution_summary = summarize(distribution, ["diffusion", "cpa", "chemcpa"], DISTRIBUTION_METRICS)
    write_xlsx(PREDICTION_SUMMARY_XLSX, summary_to_rows(prediction_summary), "metrics")
    write_xlsx(DISTRIBUTION_SUMMARY_XLSX, summary_to_rows(distribution_summary), "distribution_metrics")

    changes = []
    old_pred_rows = cpa_summary_row(old_prediction, "CPA", PREDICTION_METRICS, "prediction")
    new_pred_rows = cpa_summary_row(prediction, "CPA", PREDICTION_METRICS, "prediction")
    old_dist_rows = cpa_summary_row(old_distribution, "cpa", DISTRIBUTION_METRICS, "distribution")
    new_dist_rows = cpa_summary_row(distribution, "cpa", DISTRIBUTION_METRICS, "distribution")
    for before, after in zip(old_pred_rows + old_dist_rows, new_pred_rows + new_dist_rows):
        metric = before["metric"]
        before_mean = before["mean"]
        after_mean = after["mean"]
        delta = after_mean - before_mean
        percent_change = delta / before_mean * 100 if before_mean else float("nan")
        improvement_percent = -percent_change if metric in LOWER_IS_BETTER else percent_change
        changes.append(
            {
                "group": before["group"],
                "metric": metric,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "delta": delta,
                "percent_change": percent_change,
                "improvement_percent": improvement_percent,
                "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                "before_std": before["std"],
                "after_std": after["std"],
            }
        )
    change_df = pd.DataFrame(changes)
    change_df.to_csv(CHANGE_SUMMARY_CSV, index=False)
    plot_changes(change_df)

    print("prediction CPA rows:", int(prediction_mask.sum()), "matched:", int(prediction_matches.sum()))
    print("distribution cpa rows:", int(distribution_mask.sum()), "matched:", int(distribution_matches.sum()))
    print("wrote:", PREDICTION_OUTPUT)
    print("wrote:", DISTRIBUTION_OUTPUT)
    print("wrote:", PREDICTION_SUMMARY_XLSX)
    print("wrote:", DISTRIBUTION_SUMMARY_XLSX)
    print("wrote:", CHANGE_SUMMARY_CSV)
    print("wrote:", PLOT_OUTPUT)
    print(change_df.to_string(index=False))


if __name__ == "__main__":
    main()
