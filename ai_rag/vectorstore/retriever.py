import logging
import math
import re
from collections import Counter
from typing import Any, Iterable, List, Tuple

from langchain_core.documents import Document


logger = logging.getLogger(__name__)

RRF_K = 60
KEYWORD_TOP_K_MULTIPLIER = 4
MIN_KEYWORD_TOKEN_LENGTH = 2

DOMAIN_KEYWORD_BOOSTS = {
    "불용": ("불용", "사용중단", "불용결정"),
    "반납": ("반납", "반환"),
    "관리전환": ("관리전환", "관리 전환"),
    "내용연수": ("내용연수", "내용 연수", "사용주기", "사용 주기"),
    "g2b": ("g2b", "G2B", "목록번호", "목록 번호"),
    "취득": ("취득", "구매", "검수"),
    "등록": ("등록", "대장등록", "물품등록"),
    "처분": ("처분", "매각", "폐기", "양여", "해체"),
    "AI 챗봇": ("챗봇", "쳇봇", "AI 챗봇", "FAQ", "질문 예시", "답변 가능", "챗봇 기능"),
    "보유현황": ("보유현황", "보유 현황", "현황 조회", "재고", "통계"),
    "식별": ("물품고유번호", "G2B", "목록번호", "식별번호", "분류번호"),
    "절차": ("절차", "방법", "단계", "승인", "확정", "취소", "수정"),
}

CONFUSION_PAIRS = (
    ("반납", "불용"),
    ("취득", "등록"),
    ("처분", "불용"),
)

COMPARISON_TERMS = ("차이", "비교", "구분", "vs", "VS", "다른", "같은")


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower())
    return [token for token in tokens if len(token) >= MIN_KEYWORD_TOKEN_LENGTH]


def _doc_key(doc: Document) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(
        metadata.get("doc_id")
        or (metadata.get("source"), metadata.get("chapter"), metadata.get("title"), metadata.get("chunk_index"))
        or id(doc)
    )


def _metadata_text(metadata: dict[str, Any]) -> str:
    fields = ("doc_type", "source", "chapter", "category", "title", "section_path", "question")
    return " ".join(str(metadata.get(field) or "") for field in fields)


def _document_search_text(doc: Document) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return f"{_metadata_text(metadata)} {getattr(doc, 'page_content', '') or ''}"


def _get_all_documents(vectordb, metadata_filter: dict[str, Any] | None = None) -> list[Document]:
    if not hasattr(vectordb, "get"):
        return []

    get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if metadata_filter:
        get_kwargs["where"] = metadata_filter

    raw = vectordb.get(**get_kwargs)
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    ids = raw.get("ids") or []

    docs: list[Document] = []
    for index, page_content in enumerate(documents):
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        if ids and index < len(ids) and not metadata.get("doc_id"):
            metadata["doc_id"] = ids[index]
        docs.append(Document(page_content=page_content or "", metadata=metadata))
    return docs


def _bm25_keyword_search(query: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
    query_tokens = _tokenize(query)
    if not query_tokens or not docs:
        return []

    tokenized_docs = [_tokenize(_document_search_text(doc)) for doc in docs]
    doc_count = len(tokenized_docs)
    avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    k1 = 1.5
    b = 0.75
    scored: list[tuple[Document, float]] = []
    for doc, doc_tokens in zip(docs, tokenized_docs):
        if not doc_tokens:
            continue

        frequencies = Counter(doc_tokens)
        score = 0.0
        doc_len = len(doc_tokens)
        for token in query_tokens:
            tf = frequencies.get(token, 0)
            if not tf:
                continue
            idf = math.log(1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denominator = tf + k1 * (1 - b + b * doc_len / max(avg_len, 1))
            score += idf * (tf * (k1 + 1) / denominator)

        score += _exact_domain_keyword_score(query, doc)
        if score > 0:
            scored.append((doc, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def _matched_domain_keywords(query: str) -> set[str]:
    normalized_query = str(query or "").lower()
    matched = set()
    for canonical, aliases in DOMAIN_KEYWORD_BOOSTS.items():
        if any(alias.lower() in normalized_query for alias in aliases):
            matched.add(canonical)
    return matched


def _is_comparison_query(query: str) -> bool:
    return any(term in str(query or "") for term in COMPARISON_TERMS)


def _exact_domain_keyword_score(query: str, doc: Document) -> float:
    matched = _matched_domain_keywords(query)
    if not matched:
        return 0.0

    metadata = getattr(doc, "metadata", {}) or {}
    search_text = f"{_metadata_text(metadata)} {getattr(doc, 'page_content', '') or ''}".lower()
    score = 0.0
    for canonical in matched:
        aliases = DOMAIN_KEYWORD_BOOSTS[canonical]
        if any(alias.lower() in search_text for alias in aliases):
            score += 0.35
    return score


def _metadata_domain_boost(query: str, doc: Document) -> tuple[float, list[str]]:
    matched = _matched_domain_keywords(query)
    if not matched:
        return 0.0, []

    metadata = getattr(doc, "metadata", {}) or {}
    metadata_text = _metadata_text(metadata).lower()
    boost = 0.0
    reasons = []
    for canonical in matched:
        aliases = DOMAIN_KEYWORD_BOOSTS[canonical]
        if any(alias.lower() in metadata_text for alias in aliases):
            boost += 0.012
            reasons.append(f"metadata:{canonical}")

    if metadata.get("doc_type") == "manual_chunk" and boost:
        boost += 0.004
        reasons.append("manual_chunk")
    return boost, reasons


def _negative_hint(query: str, doc: Document) -> tuple[float, list[str]]:
    if _is_comparison_query(query):
        return 0.0, []

    matched = _matched_domain_keywords(query)
    if not matched:
        return 0.0, []

    metadata_text = _metadata_text(getattr(doc, "metadata", {}) or {}).lower()
    penalty = 0.0
    reasons = []
    for left, right in CONFUSION_PAIRS:
        if left in matched and right not in matched:
            aliases = DOMAIN_KEYWORD_BOOSTS[right]
            if any(alias.lower() in metadata_text for alias in aliases):
                penalty -= 0.014
                reasons.append(f"negative:{right}")
        if right in matched and left not in matched:
            aliases = DOMAIN_KEYWORD_BOOSTS[left]
            if any(alias.lower() in metadata_text for alias in aliases):
                penalty -= 0.014
                reasons.append(f"negative:{left}")
    return penalty, reasons


def _rank_map(results: Iterable[tuple[Document, Any]]) -> dict[str, int]:
    ranks = {}
    for rank, (doc, _score) in enumerate(results, start=1):
        ranks.setdefault(_doc_key(doc), rank)
    return ranks


def _rrf_fuse(
    dense_results: list[tuple[Document, float]],
    keyword_results: list[tuple[Document, float]],
    query: str,
    top_k: int,
) -> list[tuple[Document, float]]:
    dense_ranks = _rank_map(dense_results)
    keyword_ranks = _rank_map(keyword_results)
    docs_by_key: dict[str, Document] = {}

    for doc, _score in keyword_results:
        docs_by_key[_doc_key(doc)] = doc
    for doc, _score in dense_results:
        docs_by_key[_doc_key(doc)] = doc

    fused = []
    max_fusion_score = 0.0
    for key, doc in docs_by_key.items():
        rrf_score = 0.0
        if key in dense_ranks:
            rrf_score += 1 / (RRF_K + dense_ranks[key])
        if key in keyword_ranks:
            rrf_score += 1 / (RRF_K + keyword_ranks[key])

        boost, boost_reasons = _metadata_domain_boost(query, doc)
        penalty, penalty_reasons = _negative_hint(query, doc)
        fusion_score = max(rrf_score + boost + penalty, 0.0)
        max_fusion_score = max(max_fusion_score, fusion_score)
        fused.append((doc, fusion_score, rrf_score, boost, penalty, boost_reasons + penalty_reasons))

    if not fused:
        return dense_results

    fused.sort(key=lambda item: item[1], reverse=True)
    max_fusion_score = max(max_fusion_score, 1e-9)

    results = []
    for doc, fusion_score, rrf_score, boost, penalty, reasons in fused[:top_k]:
        metadata = getattr(doc, "metadata", {}) or {}
        
        # 1. 반드시 여기서 '먼저' 계산을 해야 합니다!
        distance_like_score = 1 - (fusion_score / max_fusion_score)
        
        metadata["_retrieval"] = {
            "method": "hybrid_rrf",
            "dense_rank": dense_ranks.get(_doc_key(doc)),
            "keyword_rank": keyword_ranks.get(_doc_key(doc)),
            "rrf_score": round(rrf_score, 6),
            "metadata_boost": round(boost, 6),
            "negative_hint": round(penalty, 6),
            "hints": reasons,
        }
        
        # 2. 위에서 계산된 값을 여기에 집어넣습니다!
        metadata["hybrid_rrf_score"] = round(rrf_score, 6)
        metadata["hybrid_normalized_score"] = max(distance_like_score, 0.0)
        
        doc.metadata = metadata
        results.append((doc, max(distance_like_score, 0.0)))
        
    return results


def retrieve_docs(
    vectordb,
    query: str,
    top_k: int,
    metadata_filter: dict[str, Any] | None = None,
    use_hybrid: bool = True,
) -> List[Tuple]:
    """
    Chroma dense 검색과 BM25/키워드 검색을 함께 수행하고 RRF로 병합한다.

    반환값은 [(Document, score), ...] 형식이다. hybrid score는 기존 chain의
    필터와 호환되도록 낮을수록 좋은 distance-like 점수로 정규화한다.
    """
    search_kwargs = {"query": query, "k": top_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    dense_results = vectordb.similarity_search_with_score(**search_kwargs)
    
    for doc, score in dense_results:
        metadata = getattr(doc, "metadata", {}) or {}
        metadata["raw_dense_score"] = score
        doc.metadata = metadata
    
    if not use_hybrid:
        logger.info(
            "[Retriever] query=%r top_k=%d filter=%s method=dense_only returned=%d",
            query[:120],
            top_k,
            metadata_filter,
            len(dense_results),
        )
        return dense_results

    corpus_docs = _get_all_documents(vectordb, metadata_filter=metadata_filter)
    keyword_results = _bm25_keyword_search(
        query=query,
        docs=corpus_docs,
        top_k=max(top_k * KEYWORD_TOP_K_MULTIPLIER, top_k),
    )
    results = _rrf_fuse(dense_results, keyword_results, query=query, top_k=top_k)

    if results:
        best_score = results[0][1]
        logger.info(
            "[Retriever] query=%r top_k=%d filter=%s dense=%d keyword=%d returned=%d best_score=%.4f",
            query[:120],
            top_k,
            metadata_filter,
            len(dense_results),
            len(keyword_results),
            len(results),
            best_score,
        )
    else:
        logger.info(
            "[Retriever] query=%r top_k=%d filter=%s dense=%d keyword=%d returned=0",
            query[:120],
            top_k,
            metadata_filter,
            len(dense_results),
            len(keyword_results),
        )

    return results
