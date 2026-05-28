from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import chromadb
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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


def _load_config_defaults() -> dict:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from app import config

        return {
            "retriever_top_k": int(config.RETRIEVER_TOP_K),
            "rerank_candidate_k": int(config.RERANK_CANDIDATE_K),
            "rerank_top_n": int(config.RERANK_TOP_N),
            "similarity_threshold": float(config.SIMILARITY_SCORE_THRESHOLD),
            "use_reranking": bool(config.USE_RERANKING),
        }
    except Exception:
        return {
            "retriever_top_k": 25,
            "rerank_candidate_k": 15,
            "rerank_top_n": 10,
            "similarity_threshold": 10.0,
            "use_reranking": True,
        }


def _load_chroma_embeddings(db_path: Path, collection_name: str) -> tuple[np.ndarray, list[str], list[dict]]:
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(collection_name)
    payload = collection.get(include=["embeddings", "documents", "metadatas"])

    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    documents = payload["documents"]
    metadatas = payload["metadatas"]

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"No embeddings found in {db_path!s}/{collection_name}")

    return embeddings, documents, metadatas


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _nearest_neighbor_metrics(
    embeddings: np.ndarray,
    labels: dict[str, list[str]],
    ks: list[int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    normalized = _l2_normalize(embeddings)
    cosine_similarity = normalized @ normalized.T
    np.fill_diagonal(cosine_similarity, -np.inf)

    order = np.argsort(-cosine_similarity, axis=1)
    max_k = max(ks)
    top_indices = order[:, :max_k]

    metric_rows = []
    for k in ks:
        row = {"k": k}
        neighbors = top_indices[:, :k]
        for label_name, values in labels.items():
            values_array = np.asarray(values, dtype=object)
            matches = values_array[neighbors] == values_array[:, None]
            row[f"{label_name}_hit_at_k"] = float(matches.any(axis=1).mean())
            row[f"{label_name}_precision_at_k"] = float(matches.mean())
        metric_rows.append(row)

    diagnostics = {
        "normalized_embeddings": normalized,
        "cosine_similarity": cosine_similarity,
        "top_indices": top_indices,
    }
    return pd.DataFrame(metric_rows), diagnostics


def _category_breakdown(
    categories: list[str],
    top_indices: np.ndarray,
    k: int = 5,
) -> pd.DataFrame:
    categories_array = np.asarray(categories, dtype=object)
    rows = []

    for category in sorted(set(categories)):
        query_mask = categories_array == category
        if query_mask.sum() < 2:
            continue

        neighbor_indices = top_indices[query_mask, :k]
        matches = categories_array[neighbor_indices] == category
        rows.append(
            {
                "category": category,
                "documents": int(query_mask.sum()),
                f"precision_at_{k}": float(matches.mean()),
                f"hit_at_{k}": float(matches.any(axis=1).mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["documents", "category"], ascending=[False, True])


def _silhouette_by_category(embeddings: np.ndarray, categories: list[str]) -> float | None:
    counts = Counter(categories)
    usable_mask = np.asarray([counts[c] > 1 for c in categories])
    usable_categories = [c for c, ok in zip(categories, usable_mask) if ok]

    if len(set(usable_categories)) < 2:
        return None

    return float(
        silhouette_score(
            embeddings[usable_mask],
            np.asarray(usable_categories, dtype=object),
            metric="cosine",
        )
    )


def _plot_retrieval_curve(metrics: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "01_retrieval_consistency_at_k.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(
        metrics["k"],
        metrics["category_hit_at_k"] * 100,
        marker="o",
        linewidth=2.5,
        label="Category Hit@k",
        color="#1F77B4",
    )
    ax.plot(
        metrics["k"],
        metrics["category_precision_at_k"] * 100,
        marker="s",
        linewidth=2.5,
        label="Category Precision@k",
        color="#2CA02C",
    )
    ax.plot(
        metrics["k"],
        metrics["chapter_hit_at_k"] * 100,
        marker="^",
        linewidth=2.5,
        label="Chapter Hit@k",
        color="#D62728",
    )

    ax.set_title("RAG Retrieval Consistency by Top-k", fontsize=14, pad=12)
    ax.set_xlabel("Retrieved documents (k)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_category_composition(categories: list[str], output_dir: Path) -> Path:
    path = output_dir / "02_knowledge_base_category_composition.png"
    counts = Counter(categories)
    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    names = [item[0] for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(names, values, color="#4C78A8")
    ax.invert_yaxis()
    ax.set_title("Knowledge Base Coverage by Category", fontsize=14, pad=12)
    ax.set_xlabel("Number of embedded QA documents")
    ax.grid(axis="x", alpha=0.22)
    ax.bar_label(bars, padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_similarity_distribution(
    cosine_similarity: np.ndarray,
    categories: list[str],
    output_dir: Path,
) -> Path:
    path = output_dir / "03_semantic_distance_distribution.png"
    categories_array = np.asarray(categories, dtype=object)
    best_same = []
    best_different = []

    for i, category in enumerate(categories_array):
        same_mask = categories_array == category
        different_mask = categories_array != category
        same_mask[i] = False

        if same_mask.any():
            best_same.append(float(np.max(cosine_similarity[i, same_mask])))
        if different_mask.any():
            best_different.append(float(np.max(cosine_similarity[i, different_mask])))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(
        1 - np.asarray(best_same),
        bins=22,
        alpha=0.72,
        label="Nearest same-category document",
        color="#2CA02C",
    )
    ax.hist(
        1 - np.asarray(best_different),
        bins=22,
        alpha=0.58,
        label="Nearest other-category document",
        color="#FF7F0E",
    )
    ax.set_title("Semantic Distance Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Cosine distance (lower is closer)")
    ax.set_ylabel("Number of QA documents")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_embedding_pca(
    normalized_embeddings: np.ndarray,
    categories: list[str],
    output_dir: Path,
) -> Path:
    path = output_dir / "04_embedding_space_pca.png"
    coords = PCA(n_components=2, random_state=42).fit_transform(normalized_embeddings)
    categories_array = np.asarray(categories, dtype=object)
    unique_categories = sorted(set(categories))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for idx, category in enumerate(unique_categories):
        mask = categories_array == category
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=26,
            alpha=0.8,
            label=category,
            color=cmap(idx % 10),
            edgecolors="white",
            linewidths=0.35,
        )

    ax.set_title("2D Projection of RAG Embeddings", fontsize=14, pad=12)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pipeline_funnel(config: dict, output_dir: Path) -> Path:
    path = output_dir / "05_rag_pipeline_funnel.png"

    if config["use_reranking"]:
        stages = [
            ("Vector retrieval", config["retriever_top_k"]),
            ("Threshold filter", config["retriever_top_k"]),
            ("Rerank candidates", config["rerank_candidate_k"]),
            ("Final context", config["rerank_top_n"]),
        ]
    else:
        stages = [
            ("Vector retrieval", config["retriever_top_k"]),
            ("Threshold filter", config["retriever_top_k"]),
            ("Final context", config["rerank_top_n"]),
        ]

    labels = [stage[0] for stage in stages]
    values = [stage[1] for stage in stages]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"][: len(stages)]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("RAG Context Selection Pipeline", fontsize=14, pad=12)
    ax.set_ylabel("Documents per query")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", alpha=0.22)
    ax.bar_label(bars, padding=4, fontsize=10)
    note = (
        f"Similarity threshold: {config['similarity_threshold']}, "
        f"reranking: {'on' if config['use_reranking'] else 'off'}"
    )
    ax.text(0.5, -0.18, note, transform=ax.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_summary(
    output_dir: Path,
    metrics: pd.DataFrame,
    category_metrics: pd.DataFrame,
    categories: list[str],
    sources: list[str],
    silhouette: float | None,
    plot_paths: list[Path],
    config: dict,
) -> None:
    summary = {
        "documents": len(categories),
        "categories": len(set(categories)),
        "sources": len(set(sources)),
        "silhouette_cosine_by_category": silhouette,
        "retriever_config": config,
        "top_k_metrics": metrics.to_dict(orient="records"),
        "category_precision_at_5": category_metrics.to_dict(orient="records"),
        "plots": [str(path) for path in plot_paths],
    }

    (output_dir / "rag_panel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best = metrics.set_index("k")
    lines = [
        "# RAG panel metric summary",
        "",
        f"- Embedded QA documents: {len(categories)}",
        f"- Knowledge categories: {len(set(categories))}",
        f"- Source manual files: {len(set(sources))}",
        f"- Category Hit@1: {best.loc[1, 'category_hit_at_k'] * 100:.1f}%",
        f"- Category Hit@5: {best.loc[5, 'category_hit_at_k'] * 100:.1f}%",
        f"- Category Precision@5: {best.loc[5, 'category_precision_at_k'] * 100:.1f}%",
        f"- Chapter Hit@5: {best.loc[5, 'chapter_hit_at_k'] * 100:.1f}%",
    ]
    if silhouette is not None:
        lines.append(f"- Category silhouette score: {silhouette:.3f}")
    lines.extend(["", "## Plot files", *[f"- {path.name}" for path in plot_paths]])

    (output_dir / "rag_panel_summary.md").write_text("\n".join(lines), encoding="utf-8")
    metrics.to_csv(output_dir / "rag_top_k_metrics.csv", index=False, encoding="utf-8-sig")
    category_metrics.to_csv(
        output_dir / "rag_category_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate panel-ready RAG retrieval metrics and plots from a Chroma DB."
    )
    parser.add_argument("--db-path", type=Path, default=_repo_root() / "chroma_db")
    parser.add_argument("--collection", default="langchain")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "ai_rag" / "results" / "rag_panel",
    )
    args = parser.parse_args()

    _configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, _documents, metadatas = _load_chroma_embeddings(args.db_path, args.collection)
    categories = [metadata.get("category") or "Unknown" for metadata in metadatas]
    chapters = [metadata.get("chapter") or "Unknown" for metadata in metadatas]
    sources = [metadata.get("source") or "Unknown" for metadata in metadatas]

    ks = [1, 3, 5, 10, 25]
    metrics, diagnostics = _nearest_neighbor_metrics(
        embeddings=embeddings,
        labels={"category": categories, "chapter": chapters, "source": sources},
        ks=ks,
    )
    category_metrics = _category_breakdown(categories, diagnostics["top_indices"], k=5)
    silhouette = _silhouette_by_category(diagnostics["normalized_embeddings"], categories)
    config = _load_config_defaults()

    plot_paths = [
        _plot_retrieval_curve(metrics, args.output_dir),
        _plot_category_composition(categories, args.output_dir),
        _plot_similarity_distribution(diagnostics["cosine_similarity"], categories, args.output_dir),
        _plot_embedding_pca(diagnostics["normalized_embeddings"], categories, args.output_dir),
        _plot_pipeline_funnel(config, args.output_dir),
    ]

    _write_summary(
        output_dir=args.output_dir,
        metrics=metrics,
        category_metrics=category_metrics,
        categories=categories,
        sources=sources,
        silhouette=silhouette,
        plot_paths=plot_paths,
        config=config,
    )

    print(f"Generated {len(plot_paths)} plots in {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
