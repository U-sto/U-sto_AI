from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configure_matplotlib() -> None:
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans KR",
        "AppleGothic",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]
    selected_font = next((font for font in preferred_fonts if font in installed_fonts), "DejaVu Sans")

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["font.family"] = selected_font


def _load_config_defaults() -> dict[str, Any]:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from app import config

        return {
            "generation_model": str(config.LLM_MODEL_NAME),
            "judge_model": str(config.LLM_MODEL_NAME),
            "embedding_model": str(config.EMBEDDING_MODEL_NAME),
            "top_k": min(int(getattr(config, "RERANK_TOP_N", 5)), 10),
        }
    except Exception:
        return {
            "generation_model": "gpt-4o",
            "judge_model": "gpt-4o",
            "embedding_model": "text-embedding-3-small",
            "top_k": 5,
        }


def _load_qa_samples(dataset_path: Path, sample_size: int | None, seed: int) -> list[dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}")

    samples = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if item.get("question") and item.get("answer"):
            sample = dict(item)
            sample["eval_id"] = f"qa_{idx:03d}"
            samples.append(sample)

    if not samples:
        raise ValueError(f"No QA samples found in {dataset_path}")

    if sample_size is not None and sample_size < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, sample_size)
        samples.sort(key=lambda item: item["eval_id"])

    return samples


def _load_chroma_documents(db_path: Path, collection_name: str) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(collection_name)
    payload = collection.get(include=["embeddings", "documents", "metadatas"])

    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    documents = list(payload["documents"])
    metadatas = list(payload["metadatas"])

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"No embeddings found in {db_path!s}/{collection_name}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms, documents, metadatas


def _retry(callable_obj, retries: int = 3, delay_seconds: float = 2.0):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return callable_obj()
        except Exception as exc:  # OpenAI errors vary by package version.
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds * attempt)
    raise last_error


def _embed_questions(client: OpenAI, questions: list[str], model: str) -> np.ndarray:
    response = _retry(lambda: client.embeddings.create(model=model, input=questions))
    vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
    embeddings = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _retrieve_context(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    documents: list[str],
    metadatas: list[dict[str, Any]],
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    similarities = document_embeddings @ query_embedding
    top_indices = np.argsort(-similarities)[:top_k]

    context_blocks = []
    retrieved = []
    for rank, doc_idx in enumerate(top_indices, start=1):
        metadata = metadatas[int(doc_idx)] or {}
        score = float(similarities[int(doc_idx)])
        title = metadata.get("title", "")
        source = metadata.get("source", "")
        chapter = metadata.get("chapter", "")
        category = metadata.get("category", "")
        document_text = documents[int(doc_idx)] or ""

        context_blocks.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"source={source}, chapter={chapter}, category={category}, title={title}, similarity={score:.4f}",
                    document_text,
                ]
            )
        )
        retrieved.append(
            {
                "rank": rank,
                "similarity": score,
                "source": source,
                "chapter": chapter,
                "category": category,
                "title": title,
            }
        )

    return "\n\n".join(context_blocks), retrieved


def _chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    response_format: dict[str, str] | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        response = _retry(lambda: client.chat.completions.create(**kwargs))
    except Exception:
        if "response_format" not in kwargs:
            raise
        kwargs.pop("response_format")
        response = _retry(lambda: client.chat.completions.create(**kwargs))

    return response.choices[0].message.content or ""


def _generate_rag_answer(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    temperature: float,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are the answer generator for a Korean university asset-management RAG chatbot. "
                "Answer in Korean. Use only the provided context. If the context does not support an answer, "
                "say that the provided documents are insufficient instead of guessing."
            ),
        },
        {
            "role": "user",
            "content": f"[Retrieved context]\n{context}\n\n[Question]\n{question}",
        },
    ]
    return _chat_completion(client, model, messages, temperature=temperature)


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _coerce_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, score))


def _judge_answer(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    generated_answer: str,
    reference_answer: str,
) -> dict[str, Any]:
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
                "You are a strict evaluator for a RAG system. Return only valid JSON. "
                "Evaluate the generated answer against the retrieved context and the user question. "
                "Faithfulness means factual claims in the answer are supported by the retrieved context. "
                "Answer relevance means the answer directly addresses the question. "
                "Reference alignment means the answer covers the core facts in the reference answer. "
                "Hallucination is true if the answer contains unsupported or contradictory factual claims. "
                "Do not reward unsupported facts just because they appear in the reference answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Return JSON with this schema:\n{json.dumps(schema_hint, ensure_ascii=False)}\n\n"
                f"[Question]\n{question}\n\n"
                f"[Retrieved context]\n{context}\n\n"
                f"[Generated answer]\n{generated_answer}\n\n"
                f"[Reference answer]\n{reference_answer}"
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

    faithfulness = _coerce_score(parsed.get("faithfulness_score"))
    relevance = _coerce_score(parsed.get("answer_relevance_score"))
    reference_alignment = _coerce_score(parsed.get("reference_alignment_score"))
    hallucination = bool(parsed.get("hallucination"))

    unsupported_claims = parsed.get("unsupported_claims")
    if not isinstance(unsupported_claims, list):
        unsupported_claims = []

    return {
        "faithfulness_score": faithfulness,
        "answer_relevance_score": relevance,
        "reference_alignment_score": reference_alignment,
        "hallucination": hallucination,
        "unsupported_claims": unsupported_claims,
        "faithfulness_rationale": str(parsed.get("faithfulness_rationale", "")),
        "answer_relevance_rationale": str(parsed.get("answer_relevance_rationale", "")),
        "reference_alignment_rationale": str(parsed.get("reference_alignment_rationale", "")),
        "judge_raw": parsed,
    }


def _load_existing_results(raw_path: Path) -> dict[str, dict[str, Any]]:
    if not raw_path.exists():
        return {}

    results = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        results[item["eval_id"]] = item
    return results


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _format_context_preview(retrieved: list[dict[str, Any]]) -> str:
    return " | ".join(
        f"{item['rank']}:{item.get('source', '')}/{item.get('chapter', '')}/{item.get('category', '')}"
        for item in retrieved
    )


def _plot_metric_bars(summary: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "01_llm_judge_metric_scores.png"
    labels = ["Faithfulness", "Answer relevance", "Reference alignment", "Non-hallucination"]
    values = [
        summary["faithfulness_mean"] / 5 * 100,
        summary["answer_relevance_mean"] / 5 * 100,
        summary["reference_alignment_mean"] / 5 * 100,
        (1 - summary["hallucination_rate"]) * 100,
    ]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("LLM Judge Evaluation of RAG Answers", fontsize=14, pad=12)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.22)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=4, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_score_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "02_llm_judge_score_distribution.png"
    score_columns = [
        ("faithfulness_score", "Faithfulness"),
        ("answer_relevance_score", "Answer relevance"),
        ("reference_alignment_score", "Reference alignment"),
    ]
    positions = np.arange(1, 6)
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for offset, (column, label) in zip([-width, 0, width], score_columns):
        counts = df[column].value_counts().reindex(positions, fill_value=0)
        ax.bar(positions + offset, counts, width=width, label=label)

    ax.set_title("LLM Judge Score Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Judge score (1=low, 5=high)")
    ax.set_ylabel("Number of evaluated answers")
    ax.set_xticks(positions)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hallucination_by_category(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "03_hallucination_rate_by_category.png"
    grouped = (
        df.groupby("category")
        .agg(samples=("eval_id", "count"), hallucination_rate=("hallucination", "mean"))
        .reset_index()
        .sort_values(["hallucination_rate", "samples"], ascending=[False, False])
    )
    grouped["label"] = grouped["category"] + " (n=" + grouped["samples"].astype(str) + ")"

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(grouped["label"], grouped["hallucination_rate"] * 100, color="#E15759")
    ax.invert_yaxis()
    ax.set_title("Hallucination Rate by QA Category", fontsize=14, pad=12)
    ax.set_xlabel("Hallucination rate (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.22)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in grouped["hallucination_rate"] * 100], padding=3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _summarize_results(
    df: pd.DataFrame,
    output_dir: Path,
    plot_paths: list[Path],
    config: dict[str, Any],
) -> None:
    summary = {
        "sample_count": int(len(df)),
        "faithfulness_mean": float(df["faithfulness_score"].mean()),
        "answer_relevance_mean": float(df["answer_relevance_score"].mean()),
        "reference_alignment_mean": float(df["reference_alignment_score"].mean()),
        "hallucination_rate": float(df["hallucination"].mean()),
        "retrieved_source_hit_rate": float(df["retrieved_source_hit"].mean()),
        "retrieved_chapter_hit_rate": float(df["retrieved_chapter_hit"].mean()),
        "config": config,
        "plots": [str(path) for path in plot_paths],
    }

    (output_dir / "llm_judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# LLM judge RAG evaluation summary",
        "",
        f"- Evaluated samples: {summary['sample_count']}",
        f"- Faithfulness: {summary['faithfulness_mean']:.2f} / 5",
        f"- Answer relevance: {summary['answer_relevance_mean']:.2f} / 5",
        f"- Reference alignment: {summary['reference_alignment_mean']:.2f} / 5",
        f"- Hallucination rate: {summary['hallucination_rate'] * 100:.1f}%",
        f"- Retrieved source hit rate: {summary['retrieved_source_hit_rate'] * 100:.1f}%",
        f"- Retrieved chapter hit rate: {summary['retrieved_chapter_hit_rate'] * 100:.1f}%",
        "",
        "## Metric definitions",
        "- Faithfulness: answer claims supported by retrieved context.",
        "- Answer relevance: answer directly addresses the user question.",
        "- Hallucination rate: share of answers with unsupported or contradictory claims.",
        "- Reference alignment: coverage of the curated QA answer, used as an auxiliary quality check.",
        "",
        "## Plot files",
        *[f"- {path.name}" for path in plot_paths],
    ]
    (output_dir / "llm_judge_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _build_completed_dataframe(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results.values():
        retrieved = item.get("retrieved", [])
        retrieved_sources = {doc.get("source") for doc in retrieved}
        retrieved_chapters = {doc.get("chapter") for doc in retrieved}
        row = {
            "eval_id": item["eval_id"],
            "question": item["question"],
            "category": item.get("category", ""),
            "source": item.get("source", ""),
            "chapter": item.get("chapter", ""),
            "generated_answer": item.get("generated_answer", ""),
            "reference_answer": item.get("reference_answer", ""),
            "retrieved_preview": _format_context_preview(retrieved),
            "retrieved_source_hit": item.get("source") in retrieved_sources,
            "retrieved_chapter_hit": item.get("chapter") in retrieved_chapters,
        }
        row.update(item.get("judge", {}))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("eval_id")


def main() -> None:
    defaults = _load_config_defaults()
    parser = argparse.ArgumentParser(
        description="Generate LLM-judge faithfulness, relevance, and hallucination metrics for RAG answers."
    )
    parser.add_argument("--dataset-path", type=Path, default=_repo_root() / "dataset" / "qa_output" / "manual_qa_final.json")
    parser.add_argument("--db-path", type=Path, default=_repo_root() / "chroma_db")
    parser.add_argument("--collection", default="langchain")
    parser.add_argument("--output-dir", type=Path, default=_repo_root() / "ai_rag" / "results" / "llm_judge")
    parser.add_argument("--sample-size", type=int, default=None, help="Default evaluates every QA sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--generation-model", default=defaults["generation_model"])
    parser.add_argument("--judge-model", default=defaults["judge_model"])
    parser.add_argument("--embedding-model", default=defaults["embedding_model"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Ignore any existing JSONL results.")
    args = parser.parse_args()

    load_dotenv(_repo_root() / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")

    _configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.output_dir / "llm_judge_raw.jsonl"
    if args.force and raw_path.exists():
        raw_path.unlink()

    client = OpenAI()
    samples = _load_qa_samples(args.dataset_path, args.sample_size, args.seed)
    document_embeddings, documents, metadatas = _load_chroma_documents(args.db_path, args.collection)
    query_embeddings = _embed_questions(
        client,
        [sample["question"] for sample in samples],
        model=args.embedding_model,
    )

    completed = _load_existing_results(raw_path)
    for index, (sample, query_embedding) in enumerate(zip(samples, query_embeddings), start=1):
        if sample["eval_id"] in completed:
            print(f"[{index}/{len(samples)}] skipped {sample['eval_id']} (cached)")
            continue

        context, retrieved = _retrieve_context(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            documents=documents,
            metadatas=metadatas,
            top_k=args.top_k,
        )
        generated_answer = _generate_rag_answer(
            client=client,
            model=args.generation_model,
            question=sample["question"],
            context=context,
            temperature=args.temperature,
        )
        judge = _judge_answer(
            client=client,
            model=args.judge_model,
            question=sample["question"],
            context=context,
            generated_answer=generated_answer,
            reference_answer=sample["answer"],
        )

        result = {
            "eval_id": sample["eval_id"],
            "question": sample["question"],
            "category": sample.get("category", ""),
            "source": sample.get("source", ""),
            "chapter": sample.get("chapter", ""),
            "reference_answer": sample["answer"],
            "generated_answer": generated_answer,
            "retrieved": retrieved,
            "judge": judge,
        }
        _append_jsonl(raw_path, result)
        completed[sample["eval_id"]] = result
        print(
            f"[{index}/{len(samples)}] {sample['eval_id']} "
            f"faith={judge['faithfulness_score']} rel={judge['answer_relevance_score']} "
            f"hallucination={judge['hallucination']}"
        )

    df = _build_completed_dataframe(completed)
    if df.empty:
        raise RuntimeError("No completed LLM judge results were produced.")

    df.to_csv(args.output_dir / "llm_judge_results.csv", index=False, encoding="utf-8-sig")
    df.to_json(args.output_dir / "llm_judge_results.json", orient="records", force_ascii=False, indent=2)

    summary_for_plots = {
        "faithfulness_mean": float(df["faithfulness_score"].mean()),
        "answer_relevance_mean": float(df["answer_relevance_score"].mean()),
        "reference_alignment_mean": float(df["reference_alignment_score"].mean()),
        "hallucination_rate": float(df["hallucination"].mean()),
    }
    plot_paths = [
        _plot_metric_bars(summary_for_plots, args.output_dir),
        _plot_score_distribution(df, args.output_dir),
        _plot_hallucination_by_category(df, args.output_dir),
    ]
    _summarize_results(
        df=df,
        output_dir=args.output_dir,
        plot_paths=plot_paths,
        config={
            "dataset_path": str(args.dataset_path),
            "db_path": str(args.db_path),
            "collection": args.collection,
            "top_k": args.top_k,
            "generation_model": args.generation_model,
            "judge_model": args.judge_model,
            "embedding_model": args.embedding_model,
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
    )

    print(f"Generated LLM judge artifacts in {args.output_dir}")
    print((args.output_dir / "llm_judge_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
