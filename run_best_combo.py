#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the fixed best validation ensemble.

This does not train models and does not search weights. It reads the existing
val_predictions_best.json files from the selected runs, applies the fixed
ensemble weights, per-class biases, and the best consistency rule.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support


EVAL_FIELDS = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

FIELD_WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}

BEST_RULE = "timeline_na_all_na"

BEST_COMBO = {
    "promise_status": {
        "bias": {"Yes": 0.8, "No": 1.5},
        "sources": [
            {
                "path": "outputs_veripromise_esg_strong/promise_bert_ce_w/val_predictions_best.json",
                "weight": 0.6145808407072415,
            },
            {
                "path": "[oldv2]outputs_veripromise_esg_improved/promise_bert_seed42/val_predictions_best.json",
                "weight": 0.04925269197969967,
            },
            {
                "path": "outputs_veripromise_esg_strong/promise_macbert_ce_w/val_predictions_best.json",
                "weight": 0.2934580238885961,
            },
            {
                "path": "[oldv2]outputs_veripromise_esg_improved/promise_roberta_seed42/val_predictions_best.json",
                "weight": 0.04270844342446262,
            },
        ],
    },
    "verification_timeline": {
        "bias": {
            "already": 1.0,
            "within_2_years": 1.5,
            "between_2_and_5_years": 1.0,
            "more_than_5_years": 1.25,
            "N/A": 1.0,
        },
        "sources": [
            {
                "path": "[oldv2]outputs_veripromise_esg_improved/timeline_bert_lr3e-5_seed42/val_predictions_best.json",
                "weight": 0.05346161012868207,
            },
            {
                "path": "outputs_veripromise_esg_strong/timeline_bert_ce_w_lr2/val_predictions_best.json",
                "weight": 0.010012113974794153,
            },
            {
                "path": "[oldv2]outputs_veripromise_esg_improved/timeline_bert_lr2e-5_seed7/val_predictions_best.json",
                "weight": 0.07899801954671075,
            },
            {
                "path": "[oldv2]outputs_veripromise_esg_improved/timeline_roberta_lr2e-5_seed42/val_predictions_best.json",
                "weight": 0.010606161773015207,
            },
            {
                "path": "[oldv2]outputs_veripromise_esg_improved/timeline_bert_lr2e-5_seed42/val_predictions_best.json",
                "weight": 0.002936474922329506,
            },
            {
                "path": "outputs_veripromise_esg_strong/timeline_roberta_ce_w_lr2/val_predictions_best.json",
                "weight": 0.06386695724898751,
            },
            {
                "path": "outputs_veripromise_esg_strong/timeline_bert_ce_w_lr3/val_predictions_best.json",
                "weight": 0.015903259977471672,
            },
            {
                "path": "outputs_veripromise_esg_strong/timeline_macbert_ce_w_lr2/val_predictions_best.json",
                "weight": 0.1560226815220292,
            },
            {
                "path": "outputs_veripromise_esg_strong/timeline_bert_marked_lr2/val_predictions_best.json",
                "weight": 0.4127568516165702,
            },
            {
                "path": "outputs_veripromise_esg_sklearn/sk_timeline_linsvc_marked_char25_c1/val_predictions_best.json",
                "weight": 0.1954358692894097,
            },
        ],
    },
    "evidence_status": {
        "bias": {"Yes": 1.0, "No": 1.0, "N/A": 1.0},
        "sources": [
            {
                "path": "outputs_veripromise_esg_targeted/status_macbert_marked_lr2/val_predictions_best.json",
                "weight": 1.0,
            }
        ],
    },
    "evidence_quality": {
        "bias": {"Clear": 1.0, "Not Clear": 1.0, "Misleading": 1.0, "N/A": 3.0},
        "sources": [
            {
                "path": "outputs_veripromise_esg_targeted/quality_bert_marked_focal/val_predictions_best.json",
                "weight": 0.4,
            },
            {
                "path": "outputs_veripromise_esg_sklearn/sk_quality_logreg_fields_char25_c2/val_predictions_best.json",
                "weight": 0.6,
            },
        ],
    },
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_submission_csv(path: Path, val_data, preds_by_field):
    fieldnames = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(val_data):
            writer.writerow(
                {
                    "id": item["id"],
                    "promise_status": preds_by_field["promise_status"][idx],
                    "verification_timeline": preds_by_field["verification_timeline"][idx],
                    "evidence_status": preds_by_field["evidence_status"][idx],
                    "evidence_quality": preds_by_field["evidence_quality"][idx],
                }
            )


def align_rows(rows: List[Dict[str, Any]], val_ids: List[str], source_path: Path):
    if len(rows) != len(val_ids):
        raise ValueError(f"{source_path} row count mismatch: {len(rows)} != {len(val_ids)}")
    if all(str(row.get("id")) == val_id for row, val_id in zip(rows, val_ids)):
        return rows
    by_id = {str(row.get("id")): row for row in rows}
    if set(by_id) != set(val_ids):
        raise ValueError(f"{source_path} ids do not match validation ids")
    return [by_id[val_id] for val_id in val_ids]


def normalize_probs(probs: np.ndarray):
    sums = probs.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return probs / sums


def load_source_probs(base_dir: Path, val_ids: List[str], field: str, source: Dict[str, Any]):
    labels = EVAL_FIELDS[field]
    source_path = base_dir / source["path"]
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source prediction file: {source_path}")
    rows = align_rows(load_json(source_path), val_ids, source_path)
    prob_key = f"prob_{field}"
    if prob_key not in rows[0]:
        raise KeyError(f"{source_path} missing {prob_key}")
    probs = np.array(
        [[float(row[prob_key].get(label, 0.0)) for label in labels] for row in rows],
        dtype=np.float64,
    )
    return normalize_probs(probs)


def ensemble_field(base_dir: Path, val_ids: List[str], field: str):
    labels = EVAL_FIELDS[field]
    combo = BEST_COMBO[field]
    weights = np.array([float(source["weight"]) for source in combo["sources"]], dtype=np.float64)
    weights = weights / weights.sum()

    probs = np.zeros((len(val_ids), len(labels)), dtype=np.float64)
    for weight, source in zip(weights, combo["sources"]):
        probs += weight * load_source_probs(base_dir, val_ids, field, source)

    bias = np.array([float(combo["bias"].get(label, 1.0)) for label in labels], dtype=np.float64)
    adjusted = normalize_probs(probs * bias.reshape(1, -1))
    pred_ids = np.argmax(adjusted, axis=1)
    preds = [labels[int(idx)] for idx in pred_ids]
    return preds, adjusted


def set_one_hot(probs_by_field, field, row_idx, label):
    labels = EVAL_FIELDS[field]
    probs_by_field[field][row_idx, :] = 0.0
    probs_by_field[field][row_idx, labels.index(label)] = 1.0


def apply_best_rule(preds_by_field, probs_by_field):
    changed = 0
    if BEST_RULE != "timeline_na_all_na":
        return changed
    n = len(preds_by_field["verification_timeline"])
    for idx in range(n):
        if preds_by_field["verification_timeline"][idx] == "N/A":
            updates = {
                "promise_status": "No",
                "evidence_status": "N/A",
                "evidence_quality": "N/A",
            }
            for field, label in updates.items():
                if preds_by_field[field][idx] != label:
                    changed += 1
                preds_by_field[field][idx] = label
                set_one_hot(probs_by_field, field, idx, label)
    return changed


def field_metrics(y_true, y_pred, field):
    labels = EVAL_FIELDS[field]
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = []
    for label, pp, rr, ff, ss in zip(labels, p, r, f1, s):
        per_class.append(
            {
                "target_field": field,
                "label": label,
                "precision": float(pp),
                "recall": float(rr),
                "f1": float(ff),
                "support": int(ss),
            }
        )
    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "weighted_f1": weighted,
        "accuracy": float(accuracy),
        "score_contribution": macro * FIELD_WEIGHTS[field],
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "per_class": per_class,
    }


def write_outputs(out_dir: Path, val_data, preds_by_field, probs_by_field, rule_changes):
    combined_rows = []
    for idx, item in enumerate(val_data):
        row = dict(item)
        row["best_combo_rule"] = BEST_RULE
        for field, labels in EVAL_FIELDS.items():
            row[f"pred_{field}"] = preds_by_field[field][idx]
            row[f"prob_{field}"] = {
                label: float(value) for label, value in zip(labels, probs_by_field[field][idx].tolist())
            }
        combined_rows.append(row)

    save_json(out_dir / "best_combo_predictions.json", combined_rows)
    save_submission_csv(out_dir / "best_combo_submission.csv", val_data, preds_by_field)
    save_json(out_dir / "best_combo_config.json", {"rule": BEST_RULE, "rule_changes": rule_changes, "combo": BEST_COMBO})

    summary_rows = []
    per_class_rows = []
    report_lines = [
        "=" * 90,
        "Fixed Best Combo Validation Report",
        "=" * 90,
        f"Rule: {BEST_RULE}",
        f"Rule changes: {rule_changes}",
        "",
    ]
    total_score = 0.0
    for field in EVAL_FIELDS:
        y_true = [item[field] for item in val_data]
        y_pred = preds_by_field[field]
        metrics = field_metrics(y_true, y_pred, field)
        total_score += metrics["score_contribution"]
        report_lines.extend(
            [
                "-" * 90,
                f"{field} | Macro F1={metrics['macro_f1']:.6f} | Micro F1={metrics['micro_f1']:.6f} | Weight={FIELD_WEIGHTS[field]}",
                "-" * 90,
                metrics["report"],
                "",
            ]
        )
        summary_rows.append(
            {
                "target_field": field,
                "field_weight": FIELD_WEIGHTS[field],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["accuracy"],
                "score_contribution": metrics["score_contribution"],
            }
        )
        per_class_rows.extend(metrics["per_class"])

    summary_rows.append(
        {
            "target_field": "TOTAL",
            "field_weight": 1.0,
            "macro_f1": "",
            "micro_f1": "",
            "weighted_f1": "",
            "accuracy": "",
            "score_contribution": total_score,
        }
    )
    report_lines.insert(5, f"Combined weighted score: {total_score:.6f}")
    (out_dir / "best_combo_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    save_csv(out_dir / "best_combo_summary.csv", summary_rows)
    save_csv(out_dir / "best_combo_per_class_metrics.csv", per_class_rows)
    return total_score


def main():
    parser = argparse.ArgumentParser(description="Run fixed VeriPromiseESG best validation combo.")
    parser.add_argument("--base-dir", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--val-file", default="vpesg4k_val_1000.json")
    parser.add_argument("--output-dir", default="outputs_veripromise_esg_best_combo")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = base_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    val_data = load_json(base_dir / args.val_file)
    val_ids = [str(item["id"]) for item in val_data]

    preds_by_field = {}
    probs_by_field = {}
    for field in EVAL_FIELDS:
        preds, probs = ensemble_field(base_dir, val_ids, field)
        preds_by_field[field] = preds
        probs_by_field[field] = probs

    rule_changes = apply_best_rule(preds_by_field, probs_by_field)
    score = write_outputs(out_dir, val_data, preds_by_field, probs_by_field, rule_changes)

    print(f"Best combo score: {score:.6f}")
    print(f"Output dir: {out_dir}")
    print(f"Report: {out_dir / 'best_combo_report.txt'}")
    print(f"Predictions: {out_dir / 'best_combo_predictions.json'}")
    print(f"Submission CSV: {out_dir / 'best_combo_submission.csv'}")


if __name__ == "__main__":
    main()
