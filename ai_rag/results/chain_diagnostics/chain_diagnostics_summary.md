# Production RAG chain diagnostics

This report is generated from `run_rag_chain()` and is separate from the standalone LLM judge retrieval loop.

- Samples: 200
- Recall@5: 0.7800
- MRR: 0.7433
- nDCG@5: 0.7272
- Context precision: 0.5525
- Abstention accuracy: 0.9000
- Failure count: 115

## Top Failure Reasons

| reason | count |
|---|---:|
| final_context_contains_irrelevant_docs | 53 |
| expected_doc_not_retrieved | 25 |
| expected_abstain_but_answered | 20 |

## Score Distribution

| bucket | count | min | p50 | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| answerable_with_relevant_doc_retrieved | 155 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| answerable_only_wrong_docs_retrieved | 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| unanswerable_retrieved | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |