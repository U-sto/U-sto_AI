# LLM judge RAG evaluation summary

- Evaluated samples: 64
- Faithfulness: 4.94 / 5
- Answer relevance: 4.94 / 5
- Reference alignment: 4.92 / 5
- Hallucination rate: 1.6%
- Retrieved source hit rate: 98.4%
- Retrieved chapter hit rate: 98.4%

## Metric definitions
- Faithfulness: answer claims supported by retrieved context.
- Answer relevance: answer directly addresses the user question.
- Hallucination rate: share of answers with unsupported or contradictory claims.
- Reference alignment: coverage of the curated QA answer, used as an auxiliary quality check.

## Plot files
- 01_llm_judge_metric_scores.png
- 02_llm_judge_score_distribution.png
- 03_hallucination_rate_by_category.png