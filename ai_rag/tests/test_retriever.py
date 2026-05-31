import os
import sys
import unittest

from langchain_core.documents import Document


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vectorstore.retriever import retrieve_docs


def _doc(doc_id: str, text: str, category: str, doc_type: str = "manual_chunk") -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source": "manual.json",
            "chapter": category,
            "category": category,
            "title": category,
            "section_path": category,
        },
    )


class FakeVectorDb:
    def __init__(self, docs: list[Document], dense_order: list[str]):
        self.docs = docs
        self.dense_order = dense_order

    def similarity_search_with_score(self, query: str, k: int, filter: dict | None = None):
        filtered_docs = self._filter_docs(filter)
        docs_by_id = {doc.metadata["doc_id"]: doc for doc in filtered_docs}
        results = []
        for rank, doc_id in enumerate(self.dense_order, start=1):
            doc = docs_by_id.get(doc_id)
            if doc is not None:
                results.append((doc, 0.1 + rank * 0.01))
        return results[:k]

    def get(self, include: list[str], where: dict | None = None):
        filtered_docs = self._filter_docs(where)
        return {
            "ids": [doc.metadata["doc_id"] for doc in filtered_docs],
            "documents": [doc.page_content for doc in filtered_docs],
            "metadatas": [doc.metadata for doc in filtered_docs],
        }

    def _filter_docs(self, metadata_filter: dict | None):
        if not metadata_filter:
            return self.docs
        return [
            doc
            for doc in self.docs
            if all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
        ]


class TestHybridRetriever(unittest.TestCase):
    def test_domain_keyword_and_negative_hint_can_override_dense_confusion(self):
        docs = [
            _doc("disuse", "물품 불용 결정과 사용중단 절차", "물품 불용 관리"),
            _doc("return", "물품 반납 신청 반납 확정 반납 처리 절차", "물품 반납 관리"),
            _doc("acquire", "물품 취득 검수 등록 절차", "물품 취득 관리"),
        ]
        vectordb = FakeVectorDb(docs, dense_order=["disuse", "return", "acquire"])

        results = retrieve_docs(vectordb, "반납 절차 알려줘", top_k=3)

        self.assertEqual(results[0][0].metadata["doc_id"], "return")
        self.assertEqual(results[0][0].metadata["_retrieval"]["method"], "hybrid_rrf")
        self.assertIn("metadata:반납", results[0][0].metadata["_retrieval"]["hints"])
        disuse_doc = next(doc for doc, _score in results if doc.metadata["doc_id"] == "disuse")
        self.assertIn("negative:불용", disuse_doc.metadata["_retrieval"]["hints"])

    def test_metadata_filter_applies_to_dense_and_keyword_search(self):
        docs = [
            _doc("manual-return", "물품 반납 절차", "물품 반납 관리", doc_type="manual_chunk"),
            _doc("faq-return", "반납 FAQ", "FAQ", doc_type="faq"),
        ]
        vectordb = FakeVectorDb(docs, dense_order=["faq-return", "manual-return"])

        results = retrieve_docs(
            vectordb,
            "반납",
            top_k=5,
            metadata_filter={"doc_type": "manual_chunk"},
        )

        self.assertEqual([doc.metadata["doc_id"] for doc, _score in results], ["manual-return"])


if __name__ == "__main__":
    unittest.main()
