# DESIGN.md

## Scope note

Given the time-bounded nature of the take-home, I limited the dense model exploration to `all-MiniLM-L6-v2`, which was one of the explicitly allowed options and a strong fit for the CPU-only, low-latency constraint. I prioritized building a complete, reproducible evaluation harness and benchmarking BM25, dense, and hybrid retrieval under the required constraints over making a sweep over the different options. Same is true for the hybrid model.

If I had more time, the next comparison I would run would be against the other allowed dense encoders to test whether the current operating point is still optimal under the same latency and memory budget.

## 1. Chosen operating point

**Chosen configuration:** `dense-minilm` using `sentence-transformers/all-MiniLM-L6-v2` on CPU.

I selected the dense MiniLM configuration as the operating point because it provided the best overall quality while comfortably satisfying the take-home constraints on the machine I benchmarked on.

On the FiQA dev set, this configuration achieved:

- **Recall@10:** 0.4665
- **MRR:** 0.4611
- **Warm p50 latency:** 11.22 ms
- **Warm p95 latency:** 13.21 ms
- **Peak RAM:** 746 MB

This was the strongest retrieval-quality result among the evaluated configurations, and it remained well below both the **50 ms warm p95 latency** limit and the **2 GB serve-time RAM** limit.

I did not choose BM25 because its retrieval quality was substantially worse, especially on short and semantically phrased queries. I also did not choose the hybrid variants because, on this dataset and with the specific fusion setup I tested, they underperformed the dense retriever while adding latency and memory overhead.

---

## 2. Benchmark table

All runs below were measured on the FiQA dev set with `top_k=10`.

| Config | Recall@10 | MRR | Cold p50 (ms) | Cold p95 (ms) | Warm p50 (ms) | Warm p95 (ms) | Peak RAM (MB) | Meets constraints |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BM25 | 0.2829 | 0.2978 | 0.99 | 1.68 | 0.84 | 1.93 | 647.14 | Yes |
| dense-minilm | 0.4665 | 0.4611 | 13.44 | 27.06 | 11.22 | 13.21 | 746.04 | Yes |
| hybrid-rrf (fetch_k=100, rrf_k=60) | 0.4075 | 0.4209 | 15.09 | 18.11 | 14.46 | 17.80 | 898.27 | Yes |
| hybrid-rrf-ablation (fetch_k=200, rrf_k=60) | 0.4072 | 0.4199 | 13.58 | 25.79 | 14.68 | 21.55 | 958.82 | Yes |

### Stratified recall observations

#### Recall@10 by query length

| Config | Short (<5) | Medium (5–15) | Long (>15) |
|---|---:|---:|---:|
| BM25 | 0.1154 | 0.2816 | 0.3151 |
| dense-minilm | 0.4231 | 0.4705 | 0.4543 |
| hybrid-rrf | 0.1923 | 0.4090 | 0.4334 |
| hybrid-rrf-ablation | 0.1923 | 0.4066 | 0.4434 |

Dense MiniLM was especially better on **short queries**, which is consistent with FiQA containing many brief financial questions where exact keyword overlap is not sufficient.

#### Recall@10 by gold passage length

| Config | Top 10% longest gold docs | Rest |
|---|---:|---:|
| BM25 | 0.2792 | 0.2851 |
| dense-minilm | 0.4140 | 0.4974 |
| hybrid-rrf | 0.3935 | 0.4157 |
| hybrid-rrf-ablation | 0.3965 | 0.4135 |

Dense retrieval dropped on the longest-doc bucket relative to the rest, suggesting that broader passages are still harder to match cleanly with single-vector query encoding.

---

## 3. Cold vs. warm latency

Evaluation harness measures:

- **Cold-start latency** on the first 20 queries
- Then performs a **100-query warmup**
- Then measures **warm latency** on the remaining queries

The main cold-vs-warm delta appears in the dense and hybrid systems, not BM25.

### BM25
BM25 is nearly flat between cold and warm runs because it is a lightweight lexical scorer with minimal per-query state. Its warm p95 stayed under 2 ms.

### Dense MiniLM
Dense retrieval showed a visible cold-start penalty:

- cold p95: 27.06 ms
- warm p95: 13.21 ms

My best guess is that the very first queries are slower because they have to pay some one-time startup costs — things like loading the model, warming up the CPU cache, paging memory into RAM, and getting the core NumPy/FAISS functions properly loaded and ready. Once those initial bits are done and everything is 'warmed up', the subsequent queries run much more smoothly and consistently.

### Hybrid
Hybrid inherits both sparse and dense costs and adds fusion work, so it is slower than either individual retriever. The warm p95 remained comfortably under the target, but the hybrid path offered no latency advantage and no quality advantage over dense.

---

## 4. One counterintuitive finding

The only counterintuitive finding was that the **hybrid retriever did not outperform the dense retriever**.

My expectation going in was that BM25 and dense retrieval would have complementary strengths: BM25 would recover exact financial terminology, while the dense retriever would cover semantic paraphrases. I expected reciprocal-rank fusion to improve first-stage recall by combining both signals.

What I observed was the opposite: both hybrid configurations underperformed dense MiniLM on both Recall@10 and MRR, while also consuming more RAM and more latency.

### Best hypothesis

**My best hypothesis**

I suspect the dense retriever was already quite strong on this dataset, so adding BM25 mostly introduced noisy lexical matches rather than useful complementary signals. 

In FiQA, the queries are often short and packed with specific terminology, but the relevant passages are frequently phrased quite differently. Because of this, BM25 tends to pull in passages that have high keyword overlap but miss the actual intent. When we apply equal-weight RRF, those misleading sparse hits get enough boost to push aside better dense results.

### What we can try instead:

- **Replace fixed equal-weight RRF** with a **weighted or query-adaptive fusion**. For example:  
  `fused_score = α * dense_norm + (1-α) * bm25_norm`  
  with α around 0.6–0.7 (giving more weight to the dense scores).

- **Add a reranker** over the fused candidate set. RRF doesn’t reason about which interpretation of the query is actually correct — it just combines scores blindly. A good cross-encoder reranker (like `cross-encoder/ms-marco-MiniLM-L-6-v2`) should help clean this up and give a solid improvement, similar to what we saw in the dense-only experiments.

---

## 5. Approaches that didn’t pan out

### A. Hybrid RRF as the operating point

**What I tried:** Reciprocal-rank fusion of BM25 and dense retrieval with `fetch_k=100, rrf_k=60`.

**What I expected:** Better coverage than either standalone retriever.

**What I observed:** Hybrid improved over BM25 but still underperformed dense:
- Recall@10: 0.4075 vs. 0.4665 for dense
- MRR: 0.4209 vs. 0.4611 for dense

It also increased warm p95 latency and peak RAM.

**Why I think it failed:** The sparse side was not adding enough truly complementary candidates. In many cases it added keyword-heavy but semantically off-target passages, and the RRF fusion step was not selective enough to suppress them.

### B. Increasing hybrid candidate depth

**What I tried:** An ablation increasing hybrid `fetch_k` from 100 to 200 while keeping `rrf_k=60`.

**What I expected:** Slightly better recall because more candidates from both retrievers would be available before fusion.

**What I observed:** The larger candidate set did not help quality in any meaningful way:
- Recall@10 fell slightly from 0.4075 to 0.4072
- MRR fell slightly from 0.4209 to 0.4199
- warm p95 increased from 17.80 ms to 21.55 ms
- RAM increased from 898 MB to 959 MB

**Why I think it failed:** The extra candidates were mostly low-value tail results. The issue was not insufficient candidate depth; it was that the fusion policy was mixing in low-quality sparse evidence.

---

## 6. Trade-offs against the constraints

### If latency were cut in half
If the target moved from warm p95 ≤ 50 ms to ≤ 25 ms, my chosen dense operating point would still survive on the measured machine, since its warm p95 is 13.21 ms.

That stricter latency budget would make me even less interested in hybrid retrieval, because the hybrid variants already consume more latency without improving quality.

If the budget became even tighter than that, I would explore:
- a smaller dense model,
- more aggressive batching/caching for repeated queries,
- quantized embeddings,
- or a BM25-first candidate stage with a very lightweight rerank only if it actually improved quality under measurement.

### If I had a GPU budget
A GPU budget would change the dense side first, not the sparse side.

I would consider:
- trying a stronger sentence embedding model,
- expanding the dense model search beyond MiniLM,
- increasing candidate depth and optionally adding a reranker,
- or doing a two-stage dense retrieval + reranking pipeline.

---

## 7. Production concerns

- **Index versioning and reproducibility:** Store model name, embedding settings, corpus fingerprint, and build metadata with every dense index artifact so a serve-time index can always be traced back to an exact build.
- **Latency monitoring:** Track p50/p95 separately for cold and warm traffic and break them down by retriever type, since hybrid and dense have different performance envelopes.
- **Memory headroom:** Monitor resident set size in production and keep alerting thresholds well below the 2 GB limit to leave room for process growth and deployment variance.
- **Quality regression monitoring:** Keep a fixed evaluation slice and run it on every index rebuild or model change so Recall@10 / MRR regressions are caught before deployment.
- **Failure-mode inspection:** Log anonymized hard queries with retrieved IDs and scores so lexical-mismatch, semantic drift, and fusion-failure patterns can be reviewed and fed back into retrieval design.

---

## Benchmark environment

The reported numbers were measured on the following machine:

- **Platform:** Windows 10
- **Python:** 3.11.15
- **CPU:** 13th Gen Intel(R) Core(TM) i7-13700HX (2.10 GHz)
- **Logical cores:** 24
- **Physical cores:** 16
- **RAM:** 15.75 GB

---

## Final decision

For this constrained retrieval task, I would ship **dense MiniLM on CPU** as the production operating point.

It achieved the strongest quality, met the latency and memory constraints with margin, and was simpler than the hybrid alternatives that I tested. The main improvement I would investigate next is not larger fusion, but a better targeted second stage: either stronger dense modeling (*bge-base-en-v1.5*) or a selective reranking step (*e.g. cross-encoder/ms-marco-MiniLM-L-6-v2*) justified by measurement.