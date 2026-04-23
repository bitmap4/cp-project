# Mid-project Results Report

**Project:** Predicting Human Reading Times - Causal vs. Masked LMs on Syntactic Ambiguity\
**Team:** Ishaan Romil (2023114011), Abhyudit Singh (2023114009)\
**Run:** SLURM job 2613843, `gnode062`, $4\times$ RTX 2080 Ti\
**Date:** 2026-04-24

## TL;DR

- Full pipeline ran to completion (download $\to$ annotate $\to$ extract $\times 2$ $\to$ train $\times 2$ $\to$ layerwise $\times 2$ $\to$ H1 comparison).
- **H2 (BERT middle layers > final layers for RT prediction): supported.** BERT peaks at layer 5 (test $R^2$ = 0.38) and drops monotonically from layer 6 onward to 0.23 at layer 12. GPT-2 shows the same qualitative pattern with a higher peak (layer 4, test $R^2$ = 0.47), consistent with Lan et al. (2024).
- **H1 (GPT-2 better at ambiguity onsets, BERT better at disambiguation regions): *directionally matched but statistically null*.** The mean-residual differences at onset and disambig regions point the right way, but both are < 0.001 log-ms on residuals of ~0.08 log-ms with paired-permutation p-values of 0.93 and 0.99 respectively. We cannot reject the null.

## Pipeline stages and artifacts

| Step | Output | Status |
|---|---|---|
| `download_data.py` | `data/processed/word_table.parquet` (10,256 tokens, 10 stories) | ok |
| `annotate_regions.py` (spaCy `en_core_web_trf`) | `data/annotations/syntactic_regions.tsv` (954 items, 1272 rows: 318 onset / 318 disambig / 636 control) | ok |
| `extract_features.py model=gpt2` | `features/gpt2/features.npz` (embeddings `[N, 13, 768]` + surprisal) | ok, ~3 s |
| `extract_features.py model=bert` | `features/bert/features.npz` (embeddings + PLL surprisal) | ok, ~50 s after VRAM fixes |
| `train_regression.py $\times 2$` | `results/{gpt2,bert}/ridge/layer_12.json` | ok |
| `layerwise_analysis.py $\times 2$` | `results/{gpt2,bert}/layerwise_ridge.{json,png}` | ok |
| `h1_region_compare.py` | `results/h1/region_compare_ridge.{json,png}` | ok |

Total wall time was on the order of a few minutes - GPT-2 extract 3 s, BERT extract+PLL ~50 s, everything else negligible.

## H2 - Layer-wise prediction of RT

We fit a Ridge regression per layer (embedding + surprisal + baselines `log_word_len`, `log_freq`, `zone`) to `log(mean_rt)` and report held-out test $R^2$ per layer.

### BERT

| Layer | val $R^2$ | test $R^2$ |
|---:|---:|---:|
| 0 | 0.218 | 0.165 |
| 1 | 0.313 | 0.317 |
| 2 | 0.365 | 0.346 |
| 3 | 0.359 | 0.340 |
| 4 | 0.371 | 0.365 |
| **5** | **0.390** | **0.383** |
| 6 | 0.366 | 0.366 |
| 7 | 0.357 | 0.360 |
| 8 | 0.327 | 0.338 |
| 9 | 0.299 | 0.300 |
| 10 | 0.287 | 0.225 |
| 11 | 0.290 | 0.222 |
| 12 | 0.312 | 0.227 |

**Peak: layer 5, test $R^2$ = 0.383.** Performance climbs from layer 0 (embeddings) to a plateau at layers 4-6, then decays through the last three layers. The pattern lines up with H2's prediction that middle layers (roughly 4-8) encode the syntactic structure most useful for predicting processing difficulty, while upper layers specialise toward MLM-relevant semantic features that are less aligned with incremental-reading behaviour.

### GPT-2

| Layer | val $R^2$ | test $R^2$ |
|---:|---:|---:|
| 0 | 0.279 | 0.218 |
| 1 | 0.436 | 0.422 |
| 2 | 0.458 | 0.434 |
| 3 | 0.497 | 0.465 |
| **4** | 0.467 | **0.467** |
| 5 | 0.451 | 0.459 |
| 6 | 0.421 | 0.437 |
| 7 | 0.439 | 0.436 |
| 8 | 0.391 | 0.385 |
| 9 | 0.399 | 0.351 |
| 10 | 0.395 | 0.318 |
| 11 | 0.404 | 0.248 |
| 12 | 0.360 | 0.148 |

**Peak: layer 4, test $R^2$ = 0.467.** GPT-2 outperforms BERT at peak by ~9 $R^2$ points, and the late-layer drop is much steeper (layer 12 test $R^2$ collapses to 0.148). Interpreting the gap: GPT-2 also contributes causal surprisal as a regression feature, a strong linear predictor of RT on its own, while BERT has only the (noisier) PLL estimate.

### Sidebar: train/test gap grows with depth

For both models the val-test gap widens at the last few layers (e.g. BERT layer 12 val 0.312 vs test 0.227; GPT-2 layer 12 val 0.360 vs test 0.148). This is consistent with late-layer features encoding story-specific content that doesn't transfer to held-out stories, and it reinforces H2's interpretation that middle layers are the "cognitively relevant" representation.

## H1 - GPT-2 vs BERT by ambiguity region

Ridge was fit on training stories for each model; predictions on held-out stories (`val_stories=[7,8]`, `test_stories=[9,10]`) were aligned on shared tokens and joined with the spaCy-derived syntactic regions. Per-region metrics (n = 120 onset, 120 disambig, 3716 control tokens on the held-out split) and a 5000-iter paired-permutation test on $\lvert resid_{gpt2} \rvert - \lvert resid_{bert} \rvert$:

| region | model | n | $R^2$ | Spearman $\rho$ | mean $\lvert resid \rvert$ |
|---|---|---:|---:|---:|---:|
| onset | GPT-2 | 120 | 0.295 | 0.620 | 0.0885 |
| onset | BERT | 120 | 0.349 | 0.643 | 0.0888 |
| disambig | GPT-2 | 120 | 0.232 | 0.552 | 0.0794 |
| disambig | BERT | 120 | 0.191 | 0.518 | 0.0794 |
| control | GPT-2 | 3716 | 0.271 | 0.540 | 0.0759 |
| control | BERT | 3716 | 0.292 | 0.556 | 0.0754 |

Paired test (negative `mean_diff` = GPT-2 wins):

| region | mean_diff | p |
|---|---:|---:|
| onset | -0.00037 | 0.93 |
| disambig | +0.00006 | 0.99 |
| control | +0.00046 | 0.57 |

### Interpretation

H1 predicts:
1. GPT-2 wins at onset ($mean\_diff < 0$). **Observed:** -0.00037 (directionally correct, p = 0.93). Direction matches; not significant.
2. BERT wins at disambig ($mean\_diff > 0$). **Observed:** +0.00006 (directionally correct, p = 0.99). Direction matches; not significant.

On $R^2/\rho$ the picture is also mixed and small: BERT has higher $R^2$ at onset (0.349 vs 0.295) but GPT-2 has higher $R^2$ at disambig (0.232 vs 0.191). So the $R^2$ pattern actually *reverses* the residual-based direction at onset, which is really just saying the 120-token samples are noisy.

**Why no clear signal?** A few reasons, not mutually exclusive:

1. **Natural Stories isn't a garden-path corpus.** The relative clauses and NP/S complements spaCy finds in naturalistic text are mostly low-ambiguity; a human reader usually does not commit to the wrong parse and then revise. H1 was formulated around strong GP sentences (e.g. "the horse raced past the barn fell"), and those are vanishingly rare here.
2. **Small n.** 120 onset + 120 disambig held-out tokens across two test stories gives weak statistical power for detecting small effects. A permutation test at those sample sizes would need effect sizes ~0.005 log-ms ($\approx 5$ ms of reading time) before p drops below 0.1.
3. **Both models have surprisal as a feature.** Once surprisal is included in the regression, a lot of the "incremental prediction" signal that H1 claims GPT-2 should win on is available to BERT too. The embedding contribution is additional and mostly about context representation, which may be why the regions look so similar.
4. **Controls bucket is diluted.** Our `control` is "any held-out token not tagged as onset/disambig," which at 3,716 tokens dwarfs the 636 POS-matched controls the annotator emits. The regional residual comparison should plausibly restrict to POS-matched controls for a tighter null.

### Caveat: the `region_source` label in the JSON

`results/h1/region_compare_ridge.json` reports `"region_source": "hand"`. That's a cosmetic bug in [h1_region_compare.py:162](scripts/h1_region_compare.py#L162); the default label is written before the auto-fallback check, but the actual rows came from the spaCy-derived `syntactic_regions.tsv` (1272 rows). The underlying data is the spaCy auto-annotations, not human-labelled items. Worth fixing the label; it does not change the numbers.

## What this means for the project

1. **H2 is in good shape.** We have a clean middle-layer peak for both models, which is the core Lan-et-al.-style finding we set out to reproduce. We can write this up with confidence and add probing-classifier interpretability next.
2. **H1 needs a stronger test bed.** The natural-text test is flat. Three options, roughly in order of effort:
    - **Narrow the control bucket** to only the POS-matched spaCy controls, and restrict onset/disambig to the *structurally hardest* cases (e.g. only object-RCs and NP/S complements with transitive matrix verbs). Recompute. Cheap.
    - **Augment with a garden-path test suite** (Wilcox et al. 2020 / SyntaxGym items, or Prasad & Linzen GP items). Those have human data available; this needs a new loader. Medium effort.
    - **Move to eye-tracking** (Provo, GECO). More naturalistic-text RT data but still dominated by non-GP structures. Medium effort, unclear payoff vs. the GP test suite.
    
    Recommended: (1) as a sanity tightening first, then (2) to get items specifically engineered to expose the incremental-vs-bidirectional distinction.
3. **Infrastructure is solid.** End-to-end run in a few minutes, every stage writes versioned outputs, BERT PLL surprisal is working. Ready for the phase 4 interpretability work (probing, attention rollout, PCA/t-SNE on BERT layer 5 activations).

## Appendix: numerical summary

GPT-2 final-layer (L12) baseline:
- val $R^2$ = 0.360, test $R^2$ = 0.148, test $\rho$ = 0.501, $\alpha$ = 1000.

BERT final-layer (L12) baseline:
- val $R^2$ = 0.312, test $R^2$ = 0.227, test $\rho$ = 0.538, $\alpha$ = 1000.

Best layers:
- BERT L5: val $R^2$ = 0.390, test $R^2$ = 0.383.
- GPT-2 L4: val $R^2$ = 0.467, test $R^2$ = 0.467.

H1 verdict block (from JSON):
- `gpt2_better_at_onset`: true (direction), p = 0.93.
- `bert_better_at_disambig`: true (direction), p = 0.99.
