from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt


AI_RAG_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (AI_RAG_DIR, PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app import config
from generate_llm_judge_eval import (
    _chat_completion,
    _coerce_score,
    _configure_matplotlib,
    _extract_json_object,
    _plot_hallucination_by_category,
    _plot_score_distribution,
)


def _load_chain_rows(input_jsonl: Path, sample_size: int | None, seed: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if sample_size is not None and sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)
        rows.sort(key=lambda row: str(row.get("eval_id", "")))
    return rows


def _doc_ids(docs: list[dict[str, Any]]) -> str:
    return "|".join(str(doc.get("doc_id", "")) for doc in docs if doc.get("doc_id"))


def _judge_chain_answer(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = row.get("diagnostics") or {}
    metrics = row.get("metrics") or {}
    schema_hint = {
        "faithfulness_score": "integer 1-5",
        "answer_relevance_score": "integer 1-5",
        "reference_alignment_score": "integer 1-5",
        "hallucination": "boolean",
        "unsupported_claims": ["short unsupported claim strings"],
        "faithfulness_rationale": "short Korean explanation",
        "answer_relevance_rationale": "short Korean explanation",
        "reference_alignment_rationale": "short Korean explanation",
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator for a production RAG chain. Return only valid JSON. "
                "Evaluate the generated answer against the final RAG context, attribution, and user question. "
                "Faithfulness means factual claims in the answer are supported by the final context or tool result. "
                "Answer relevance means the answer directly addresses the question. "
                "Reference alignment means the answer covers the core facts in the reference answer when one exists. "
                "Hallucination is true if the answer contains unsupported or contradictory factual claims. "
                "If expected_abstain is true, reward a clear insufficient-evidence answer and penalize unsupported guessing."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Return JSON with this schema:\n{json.dumps(schema_hint, ensure_ascii=False)}\n\n"
                f"[Question]\n{row.get('question', '')}\n\n"
                f"[Expected abstain]\n{metrics.get('expected_abstain', False)}\n\n"
                f"[Final RAG context]\n{diagnostics.get('final_context_text', row.get('final_context_text', ''))}\n\n"
                f"[Attribution]\n{json.dumps(row.get('attribution', []), ensure_ascii=False)}\n\n"
                f"[Generated answer]\n{row.get('answer', '')}\n\n"
                f"[Reference answer]\n{row.get('reference_answer', '')}"
            ),
        },
    ]
    raw = _chat_completion(
        client,
        model,
        messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    parsed = _extract_json_object(raw)
    unsupported_claims = parsed.get("unsupported_claims")
    if not isinstance(unsupported_claims, list):
        unsupported_claims = []
    return {
        "faithfulness_score": _coerce_score(parsed.get("faithfulness_score")),
        "answer_relevance_score": _coerce_score(parsed.get("answer_relevance_score")),
        "reference_alignment_score": _coerce_score(parsed.get("reference_alignment_score")),
        "hallucination": bool(parsed.get("hallucination")),
        "unsupported_claims": unsupported_claims,
        "faithfulness_rationale": str(parsed.get("faithfulness_rationale", "")),
        "answer_relevance_rationale": str(parsed.get("answer_relevance_rationale", "")),
        "reference_alignment_rationale": str(parsed.get("reference_alignment_rationale", "")),
        "judge_raw": parsed,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        source = item.get("chain_row", {})
        diagnostics = source.get("diagnostics") or {}
        metrics = source.get("metrics") or {}
        final_context = source.get("final_context") or diagnostics.get("final_context_scores") or []
        row = {
            "eval_id": source.get("eval_id"),
            "question": source.get("question"),
            "category": source.get("category", ""),
            "classification": source.get("classification"),
            "refined_query": source.get("refined_query"),
            "final_context_doc_ids": _doc_ids(final_context),
            "attribution_doc_ids": _doc_ids(source.get("attribution") or []),
            "expected_abstain": metrics.get("expected_abstain", False),
            "abstained": metrics.get("abstained", False),
            "abstention_correct": metrics.get("abstention_correct", 0),
            "answer": source.get("answer", ""),
            "reference_answer": source.get("reference_answer", ""),
        }
        row.update(item.get("judge", {}))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("eval_id")


def _summarize(df: pd.DataFrame, output_dir: Path, plot_paths: list[Path], config_payload: dict[str, Any]) -> None:
    summary = {
        "evaluation_mode": "production_run_rag_chain_llm_judge",
        "sample_count": int(len(df)),
        "faithfulness_mean": float(df["faithfulness_score"].mean()),
        "answer_relevance_mean": float(df["answer_relevance_score"].mean()),
        "reference_alignment_mean": float(df["reference_alignment_score"].mean()),
        "hallucination_rate": float(df["hallucination"].mean()),
        "abstention_accuracy": float(df["abstention_correct"].mean()) if "abstention_correct" in df else None,
        "config": config_payload,
        "plots": [str(path) for path in plot_paths],
    }
    (output_dir / "chain_llm_judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Production chain LLM judge summary",
        "",
        "- Evaluation mode: production run_rag_chain() answers judged against final context.",
        f"- Evaluated samples: {summary['sample_count']}",
        f"- Faithfulness: {summary['faithfulness_mean']:.2f} / 5",
        f"- Answer relevance: {summary['answer_relevance_mean']:.2f} / 5",
        f"- Reference alignment: {summary['reference_alignment_mean']:.2f} / 5",
        f"- Hallucination rate: {summary['hallucination_rate'] * 100:.1f}%",
    ]
    if summary["abstention_accuracy"] is not None:
        lines.append(f"- Abstention accuracy: {summary['abstention_accuracy']:.4f}")
    lines.extend(["", "## Plot files", *[f"- {path.name}" for path in plot_paths]])
    (output_dir / "chain_llm_judge_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_production_metric_bars(summary: dict[str, Any], output_dir: Path) -> Path:
    faithfulness = round(float(summary["faithfulness_mean"]), 2)
    answer_relevance = round(float(summary["answer_relevance_mean"]), 2)
    reference_alignment = round(float(summary["reference_alignment_mean"]), 2)
    values = [
        faithfulness / 5 * 100,
        answer_relevance / 5 * 100,
        reference_alignment / 5 * 100,
        round(float(summary["hallucination_rate"]) * 100, 1),
    ]
    labels = [
        "Faithfulness",
        "Answer relevance",
        "Reference alignment",
        "Hallucination rate",
    ]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]

    abstention_accuracy = summary.get("abstention_accuracy")
    if abstention_accuracy is not None:
        values.append(float(abstention_accuracy) * 100)
        labels.append("Abstention accuracy")
        colors.append("#76B7B2")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score / rate (%)")
    ax.set_title("Production Chain LLM Judge Metrics")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2, 102),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    output_path = output_dir / "01_llm_judge_metric_scores.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "<br>").replace("\n", " ")


def _write_hallucination_reports(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    hallucinations = df[df["hallucination"].astype(bool)].copy()
    csv_path = output_dir / "chain_hallucination_cases.csv"
    md_path = output_dir / "chain_hallucination_cases.md"

    fieldnames = [
        "eval_id",
        "category",
        "faithfulness_score",
        "answer_relevance_score",
        "reference_alignment_score",
        "unsupported_claims",
        "final_context_doc_ids",
        "attribution_doc_ids",
        "question",
        "answer_preview",
        "faithfulness_rationale",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in hallucinations.to_dict(orient="records"):
            unsupported_claims = row.get("unsupported_claims")
            if isinstance(unsupported_claims, list):
                unsupported_claims_text = " | ".join(str(claim) for claim in unsupported_claims)
            else:
                unsupported_claims_text = str(unsupported_claims or "")
            writer.writerow(
                {
                    "eval_id": row.get("eval_id"),
                    "category": row.get("category"),
                    "faithfulness_score": row.get("faithfulness_score"),
                    "answer_relevance_score": row.get("answer_relevance_score"),
                    "reference_alignment_score": row.get("reference_alignment_score"),
                    "unsupported_claims": unsupported_claims_text,
                    "final_context_doc_ids": row.get("final_context_doc_ids"),
                    "attribution_doc_ids": row.get("attribution_doc_ids"),
                    "question": row.get("question"),
                    "answer_preview": str(row.get("answer", ""))[:300],
                    "faithfulness_rationale": row.get("faithfulness_rationale"),
                }
            )

    category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in df.to_dict(orient="records"):
        category = str(row.get("category") or "unknown")
        category_counts[category][1] += 1
        if bool(row.get("hallucination")):
            category_counts[category][0] += 1

    unsupported_terms = Counter()
    for value in hallucinations.get("unsupported_claims", []):
        claims = value if isinstance(value, list) else [value]
        for claim in claims:
            text = str(claim).strip()
            if text:
                unsupported_terms[text] += 1

    lines = [
        "# Production chain hallucination cases",
        "",
        f"- Hallucination count: {len(hallucinations)} / {len(df)}",
        f"- Hallucination rate: {(len(hallucinations) / max(len(df), 1)) * 100:.1f}%",
        "",
        "## Hallucination By Category",
        "",
        "| category | hallucinations | samples | rate |",
        "|---|---:|---:|---:|",
    ]
    for category, (hallucination_count, sample_count) in sorted(
        category_counts.items(),
        key=lambda item: (item[1][0] / max(item[1][1], 1), item[1][0], item[1][1]),
        reverse=True,
    ):
        if hallucination_count == 0:
            continue
        lines.append(
            f"| {_markdown_cell(category)} | {hallucination_count} | {sample_count} | "
            f"{(hallucination_count / max(sample_count, 1)) * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Frequent Unsupported Claims",
            "",
            "| unsupported claim | count |",
            "|---|---:|",
        ]
    )
    if unsupported_terms:
        for claim, count in unsupported_terms.most_common(15):
            lines.append(f"| {_markdown_cell(claim)} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| eval_id | category | faithfulness | unsupported_claims | final_context |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in hallucinations.to_dict(orient="records"):
        unsupported_claims = row.get("unsupported_claims")
        if isinstance(unsupported_claims, list):
            unsupported_claims_text = " | ".join(str(claim) for claim in unsupported_claims)
        else:
            unsupported_claims_text = str(unsupported_claims or "")
        lines.append(
            "| {eval_id} | {category} | {faithfulness} | {claims} | {context} |".format(
                eval_id=_markdown_cell(row.get("eval_id")),
                category=_markdown_cell(row.get("category")),
                faithfulness=row.get("faithfulness_score"),
                claims=_markdown_cell(unsupported_claims_text),
                context=_markdown_cell(row.get("final_context_doc_ids")),
            )
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_existing_results_dataframe(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    for column in [
        "faithfulness_score",
        "answer_relevance_score",
        "reference_alignment_score",
        "abstention_correct",
    ]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    if "hallucination" in df:
        df["hallucination"] = df["hallucination"].map(_coerce_bool)
    if "eval_id" in df:
        df = df.sort_values("eval_id")
    return df


def _write_production_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    config_payload: dict[str, Any],
) -> tuple[Path, Path, list[Path]]:
    hallucination_csv_path, hallucination_md_path = _write_hallucination_reports(df, output_dir)
    plot_summary = {
        "faithfulness_mean": float(df["faithfulness_score"].mean()),
        "answer_relevance_mean": float(df["answer_relevance_score"].mean()),
        "reference_alignment_mean": float(df["reference_alignment_score"].mean()),
        "hallucination_rate": float(df["hallucination"].mean()),
        "abstention_accuracy": float(df["abstention_correct"].mean()) if "abstention_correct" in df else None,
    }
    plot_paths = [
        _plot_production_metric_bars(plot_summary, output_dir),
        _plot_score_distribution(df, output_dir),
        _plot_hallucination_by_category(df, output_dir),
    ]
    _summarize(df, output_dir, plot_paths, config_payload)
    return hallucination_csv_path, hallucination_md_path, plot_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge production run_rag_chain() answers from chain_diagnostics.jsonl."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=AI_RAG_DIR / "results" / "chain_diagnostics" / "chain_diagnostics.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AI_RAG_DIR / "results" / "chain_llm_judge",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional existing chain_llm_judge_results.csv. If provided, regenerate plots and summaries without calling the judge LLM.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default=str(config.LLM_MODEL_NAME))
    args = parser.parse_args()

    _configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.results_csv is not None:
        df = _load_existing_results_dataframe(args.results_csv)
        hallucination_csv_path, hallucination_md_path, _plot_paths = _write_production_outputs(
            df,
            args.output_dir,
            {
                "results_csv": str(args.results_csv),
                "plot_only": True,
            },
        )
        print(f"Wrote {hallucination_csv_path}")
        print(f"Wrote {hallucination_md_path}")
        print(f"Regenerated production chain LLM judge plots in {args.output_dir}")
        print((args.output_dir / "chain_llm_judge_summary.md").read_text(encoding="utf-8"))
        return

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")

    client = OpenAI()
    chain_rows = _load_chain_rows(args.input_jsonl, args.sample_size, args.seed)

    results = []
    for index, row in enumerate(chain_rows, start=1):
        judge = _judge_chain_answer(client, args.judge_model, row)
        results.append({"chain_row": row, "judge": judge})
        print(
            f"[{index}/{len(chain_rows)}] {row.get('eval_id')} "
            f"faith={judge['faithfulness_score']} rel={judge['answer_relevance_score']} "
            f"hallucination={judge['hallucination']}"
        )

    raw_path = args.output_dir / "chain_llm_judge_raw.jsonl"
    _write_jsonl(raw_path, results)
    df = _build_dataframe(results)
    df.to_csv(args.output_dir / "chain_llm_judge_results.csv", index=False, encoding="utf-8-sig")
    df.to_json(args.output_dir / "chain_llm_judge_results.json", orient="records", force_ascii=False, indent=2)
    hallucination_csv_path, hallucination_md_path, plot_paths = _write_production_outputs(
        df,
        args.output_dir,
        {
            "input_jsonl": str(args.input_jsonl),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "judge_model": args.judge_model,
        },
    )
    print(f"Wrote {hallucination_csv_path}")
    print(f"Wrote {hallucination_md_path}")
    print(f"Generated production chain LLM judge artifacts in {args.output_dir}")
    print((args.output_dir / "chain_llm_judge_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
