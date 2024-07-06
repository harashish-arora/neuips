# SC³ / Solubility — Agent handoff summary

**Purpose of this document.** Give the next agent (or human) **immediate operational context**: what the project is, what exists on disk, what the paper contains, **benchmark design**, **metrics**, what the supervisor expects, and what is done vs open.

**Repo root assumed:** `Solubility/` (this file lives at `Solubility/summary.md`).

---

## 1. Scientific thesis (what we are building)

**SC³** (pronounced “SC cubed”; LaTeX: `SC\textsuperscript{3}`) is a **multi-solvent experimental solubility benchmark** built from **BigSolDB v2.1**, not another aqueous-only leaderboard.

**Core claims (conceptual):**

1. **Literature-aggregated data is messy** — units, copycats, stereo, extraction errors — so benchmarks must **audit** before comparing models.
2. **Labels are not noise-free** — inter-laboratory disagreement defines an **aleatoric floor**; the commonly quoted **0.6–0.8 log S** “wall” is often a **heavy-tail statistic**, not the same thing as **expected** error after proper handling (copycat merging, interpolation on thermodynamic curves).
3. **Multi-solvent regression is not i.i.d.** — aggregate RMSE/R² are **dominated by a few solvents** and **between-solvent shifts**, so the suite pushes **PS-RMSE** (equal weight per solvent) and **Z-RMSE** (error vs calibrated \(\sigma\) on tiers).

**Infrastructure angle:** SC³ is meant to ship **splits + \(\sigma\) + provenance-enough methodology** so analyses like **scaling laws**, **transfer from QM solvation**, and **interpretability** are meaningful — not just a single RMSE.

Canonical narrative draft also lives in **`broad_layout.md`** (aspirational numbers/method counts there may **differ** from frozen `paper/` tables — always reconcile against **`paper/tables/`** and **`reports/`** in `sc3_benchmark_data_curation_v2/`).

---

## 2. Dataset & benchmark pipeline (from BigSolDB → SC³)

### 2.1 Source data

- **BigSolDB v2.1** — large literature-aggregated multi-solvent solubility database (paper cites ~112k mole-fraction rows at ingestion scale).

### 2.2 Curation repository (`sc3_benchmark_data_curation_v2/`)

This is the **policy + regeneration spine** for §3–6 of the paper.

- **`DECISIONS.md`** — authoritative log of policies (canonicalisation **Option D** = preserve stereo; bad-DOI list; copycat detection stages; etc.), with **artifact paths** (`scripts/`, `reports/`, `data/interim/`).
- **Outputs** include interim CSVs, JSON reports, tier summaries, split manifests — used to justify numbers in LaTeX.

**Important:** Treat **`DECISIONS.md`** as the “why” behind §Data curation / §Aleatoric / §Benchmark / §Metrics.

### 2.3 Benchmark splits (detailed)

The construction is designed so **different test sets answer different scientific questions**. Full counts and anti-leakage matrices are in **`paper/sections/05.benchmark_design.tex`** (`tab:splits`, `tab:antileakage`).

**Training pool vs tier pool (leakage philosophy).**

- **Tier evaluation** uses **multi-source** \((\text{solute},\text{solvent})\) pairs with consensus labels and (often) per-point \(\sigma\). Any **solute that appears in any tier** is **removed entirely** from the ordinary Train / Eval / OOD pools at the **(solute, solvent, \(T\))** level. That way tier metrics measure **new solutes** against calibrated references — not memorised training molecules.

**Train / Eval (in-distribution solvent band).**

- Solvents are ranked by **frequency** in the training-eligible pool. The **top-\(K\) solvents** by row count (paper uses **25**) define the **in-distribution (ID)** region — where most experimental mass lives.
- **Train:** within each ID solvent, most **(solute, solvent)** combinations feed training (minus tier-excluded solutes).
- **Eval:** a **held-out fraction** of **solute lists per solvent** (paper uses **10%** per solvent) becomes **new pairs** that never appeared in Train — still **only ID solvents**. So **Eval** tests **interpolation on the solute axis** while solvents stay familiar.

**OOD (solvent out-of-distribution).**

- **Remaining long-tail solvents** (not in the top-\(K\) ID set) form the **OOD** test pool. By construction, models **never train on those solvents**, so they cannot rely on memorising solvent-specific offsets. **Solutes** may still have been seen in *other* solvents on Train — the stress test is **generalisation along the solvent axis** (and sparse-row behaviour on rare solvents).

**Gold / Silver / Bronze tiers (calibrated ground truth).**

- Built only from **multi-source** \((\text{solute},\text{solvent})\) pairs that survive copycat merging, Hall-of-Shame exclusions, and Apelblat fit-quality gates.
- For each pair, an **inter-group pair MAE** on interpolated curves defines how tight **independent labs** agree. **Nested** tiers:
  - **Gold:** tight agreement (threshold \(\leq 0.1\) log S in the paper’s tier table — “near \(\varepsilon_A\)” scale).
  - **Silver:** looser (\(\leq 0.2\)).
  - **Bronze:** broadest (\(\leq 0.5\)).
- **Consensus label** at each \((s,T)\): mean over contributing independence-group curves; **\(\sigma\)** = spread across groups (with a floor so tiny \(\sigma\) doesn’t dominate Z-RMSE). Some rows lack \(\sigma\) when fewer than two groups contribute at that \(T\).

**Why three axes matter.**

| Axis | Split family | What must generalise |
|------|----------------|----------------------|
| New **pairs**, seen solvents | Eval | Solute chemistry within known solvents |
| Seen solutes, **new solvents** | OOD | Solvent chemistry / representation |
| **New solutes**, consensus labels | Gold/Silver/Bronze | Prediction vs **measurement-aware** targets |

### 2.4 Metrics (detailed)

Defined and motivated in **`paper/sections/06.metrics.tex`** (`tab:metrics`, `tab:metric_domain`). Summary:

**Why not headline RMSE / R² alone**

- **Between-solvent location shift:** global log\(S\) spans many orders of magnitude across solvents; a model that **detects the solvent** can reduce RMSE dramatically **without** learning subtle solute chemistry (**dummy baseline** = predict per-solvent training mean — paper shows non-trivial aggregate \(R^2\) on ID band and **negative** \(R^2\) on OOD when forced to global mean).
- **Count domination:** a handful of high-frequency solvents dominate row counts — aggregate RMSE tracks those solvents.
- **Heavy tails:** RMSE is sensitive to outliers; **MedAE** is included as a robust companion.
- **MAPE** is rejected as headline: many rows have **|\(\log S\)|** near zero → MAPE denominator blows up (**unbounded** behaviour).

**Primary headline: PS-RMSE (per-solvent RMSE, averaged over solvents).**

- Compute RMSE **within each solvent**, then average **equally across solvents** (each solvent counts once). This **removes** both “always guess solvent mean” cheap wins on RMSE and row-count domination — closer to “honest within-solvent error.”

**Tier headline: Z-RMSE**

- On tier rows where **\(\sigma_i\)** is defined, compare prediction error **in units of the declared label uncertainty**: \(\sqrt{\frac{1}{n_\sigma}\sum ((\hat y_i-y_i)/\sigma_i)^2}\). **Z ≈ 1** means errors on the scale of irreducible spread; **Z ≫ 1** means model error dominates measurement disagreement encoded in \(\sigma\).

**Also reported (standard / diagnostic):** RMSE, MAE, MedAE on each split where applicable; \(f_{\text{aleatoric}}\)-style diagnostics appear in harness docs (`vansh/README.md`) — align wording with whatever the camera-ready table actually prints.

### 2.5 Five “pre-registered” analysis questions (paper §9)

Section **`paper/sections/09.ablations.tex`** frames **Q1–Q5** as a **reading guide** (not duplicate experiments):

| ID | Question | Where addressed in manuscript | Status note |
|----|----------|-------------------------------|-------------|
| **Q1** | Data vs representation vs model | Scaling (§9a), representation + SHAP (interpretations) | Partially — multimodal **loss** experiment explicitly **not run** (deferred). |
| **Q2** | What do models learn? | **`interpretations.tex`** | Large section (SHAP, blocks, Abraham/LSER, GCN stuff). |
| **Q3** | Transfer from adjacent chemistry | **`transfer.tex`** (CombiSolv-QM → fine-tune SC³) | Story drafted with figures; verify numbers vs latest runs. |
| **Q4** | Is failure due to lack of data? | **`09a.data_scaling.tex`** + scaling tables | Power-law asymptotes vs duplicate-triple \(\varepsilon^{(s)}\) floors. |
| **Q5** | Can foundation models win? | Mostly **future work** in §9 — **full FM sweep not claimed complete** in text. |

**Supervisor framing:** expect to emphasise **~3 solid answers** in the final rewrite while keeping Q5 honest or moving detail to appendix.

---

## 3. Paper (`paper/`) — structure, status, known issues

### 3.1 Venue & format

- **NeurIPS 2026 — Evaluations & Datasets track** (`\usepackage[eandd]{neurips_2026}`).
- **Double-blind** metadata in `paper/00.metadata.tex` (`Anonymous Author(s)`).
- Title: **“SC³: A Multi-Solvent Solubility Challenge with Calibrated Aleatoric Ground Truth”**.

### 3.2 Build chain

Entrypoint: **`paper/main.tex`**.

```text
Abstract → Introduction → §3 Data curation → §4 Aleatoric → §5 Benchmark
→ §6 Metrics → §7 Baselines → §8 Results → §9 Ablations (Q1–Q5 guide)
→ §9a Data scaling → Transfer → Interpretations → Discussion → bib
```

Bibliography: **`paper/references.bib`** (`plainnat`).

### 3.3 Section-by-section inventory

| File | Role |
|------|------|
| `sections/00.abstract.tex` | Full abstract (not placeholder). |
| `sections/01.introduction.tex` | Motivation + roadmap pointing to later §. Supervisor may still edit (“Sergei wrote intro”). |
| `sections/03.data_curation.tex` | Dense audit / canonicalisation / waterfall — **many tables**; supervisor wants **trim main → appendix**. |
| `sections/04.aleatoric_theory.tex` | Source integrity, Apelblat/van’t Hoff, \(\varepsilon_A\) framework — **multiple figures**. |
| `sections/05.benchmark_design.tex` | Tiers, splits, anti-leakage tables + `fig_tier_sigma` + placeholder split diagram. |
| `sections/06.metrics.tex` | Multimodality argument + metric suite tables + figs. |
| `sections/07.baselines.tex` | Family bullets + placeholder taxonomy figure; HP ref points to **`configs/best_hps.json`** in released code. |
| `sections/08.results.tex` | Narrative + `\input{tables/main_results}` + placeholder leaderboard fig. |
| `sections/09.ablations.tex` | Q1–Q5 guide — explicitly notes **deferred** loss ablation and **incomplete** FM sweep. |
| `sections/09a.data_scaling.tex` | Scaling curves, \(\varepsilon^{(s)}\) duplicate-triple floors, power-law tables (`scaling_*.tex`). |
| `sections/transfer.tex` | Long CombiSolv-QM transfer study + figures. |
| `sections/interpretations.tex` | Long interpretability section — **many figures** (SHAP blocks, solvent maps, GCN panels). |
| `sections/10.discussion.tex` | Short limitations (multi-source sparsity, incomplete FM protocols, harmonise \(\varepsilon\) variants). |

### 3.4 Figures & tables (volume warning)

Rough PDF figure count via `\includegraphics` is **on the order of mid-20s** plus **placeholder** figures (`\figplaceholder{...}`) in baselines / results / benchmark.

**Supervisor direction:** **shortlist** figures for main text; push the rest to **appendix** (appendix not wired as separate `\input` yet — still “planned editorial structure”).

**Data curation §** is **table-heavy** (not necessarily PDF-heavy); supervisor still called it visually busy — plan **appendix / SI**.

### 3.5 Main results table

- **`paper/tables/main_results.tex`** — wide leaderboard; caption notes **`---`** cells for **pending reruns**.
- Highlighting macros exist; **fill remaining methods** after compute finishes.

### 3.6 Known paper-quality risks (from internal review notes / `claude_chat.txt` themes)

Do **not** treat chat logs as truth — use as a **checklist**:

- **Harmonise \(\varepsilon\)** — global \(\varepsilon_A\), duplicate-split floors \(\varepsilon^{(s)}\), tier \(\sigma\) — one notation box / paragraph before submission.
- **Multi-source sparsity** — acknowledge prominently (tiers built from multi-source slice).
- **Incomplete foundation-model rows** — either finish runs or remove rows; avoid permanent `---` in camera-ready.
- **Related work** — **not** a dedicated section in `main.tex` yet; supervisor asked for **lit review or gap analysis before the big table**.

### 3.7 Recent structural edits (another agent session)

Abstract filled; Introduction roadmap added; scratch meta-lines removed from §3/§4; empty **`09a.data_scaling.tex`** populated with equations + scaling tables; **`10.discussion.tex`** added; **`references.bib`** extended (e.g. Vermeire, Lundberg SHAP, Burns/FastProp alias); Transfer cite updated to **`burns2024fastprop`**; Results/Baselines todos replaced with prose.

---

## 4. Codebase map (`Solubility/` top level)

Not every folder is equally “current”; treat **`paper/`** + **`sc3_benchmark_data_curation_v2/`** + **`vansh/`** as primary.

### 4.1 `vansh/` — unified SC³ benchmark CLI

Documented in **`vansh/README.md`**.

- **CLI:** `python sc3 cache | list | run | status | collect`.
- **~18 methods**, **5 seeds**, shared splits (`eval`, `ood`, `sc3_gold/silver/bronze`).
- **Metrics:** RMSE, MAE, R², PS-RMSE, Z-RMSE, \(f_{\text{aleatoric}}\).
- **Data path:** expects **`../sc3_benchmark_data_curation_v2/data/`** or **`SC3_DATA_DIR`**.
- **Configs:** `configs/best_hps.json`.

### 4.2 `sc3-benchmark/`

Larger benchmark workspace (see structure in **`sc3_agent_protocol.md`** — scripts, `Additional_Experiments/`, etc.). May overlap conceptually with `vansh/`; verify which tree is canonical for **your** active runs.

### 4.3 `sc3_benchmark_data_curation_v2/`

**Dataset construction + reproducibility** — scripts, `data/`, `reports/`, **`DECISIONS.md`**.

### 4.4 `Dhairya/`

Per supervisor workflow: **method-specific runners** (e.g. **`sc3-benchmark-gnn-runner/`**, descriptor runner) — self-contained experiments; data resolved via `SC3_DATA_DIR` or sibling layout.

### 4.5 `harashish/`

Often **execution variants / optimisations** (e.g. Uni-Mol batching), scripts — see **`harashish/HOW_TO_RUN.md`** / **`changes.md`**.

### 4.6 `Dissolvr/` & `molmerger-repo/`

External / legacy **solubility modelling** codebases referenced by broader project — not always the SC³ paper spine.

### 4.7 `Plan/`

Phase planning (`phase_01` … `phase_09`), **`AGENT_START_HERE.md`**, **`HANDOFF.md`**, handoff notes — coordination for multi-agent work.

### 4.8 `sc3_agent_protocol.md`

Long **research protocol** document (idealised `sc3-benchmark/` layout, phase discipline). Use as philosophy + checklist; **actual folder names on disk may differ slightly**.

### 4.9 Misc root artefacts

- **`broad_layout.md`** — narrative thesis + aspirational experiment list.
- **`claude_chat.txt`** — long mixed log (mock reviews, priorities); **do not cite as ground truth** without verifying against repo.
- **`Formatting_Instructions_For_NeurIPS_2026.zip`** — style reference.

---

## 5. Compute environment (shared server)

Documented for **`hulk`** in **`harashish/HOW_TO_RUN.md`** (paths may use `/DATATWO/users/solubility/...`).

**Typical patterns:**

- Shared venv: **`myenv/`** (large).
- GPUs: **`CUDA_VISIBLE_DEVICES`**, check **`nvidia-smi`**; use **`tmux`** for long jobs.
- Thread caps: **`OMP_NUM_THREADS`**, library-specific thread limits.

**Supervisor experiment plan (from chat):** rerun benchmark suite on updated pipeline — **~3–4 hours**, **2–3 GPUs in parallel** when possible; HP tuning reportedly **does not move headline much** — worst case reuse numbers within stated uncertainty **if** justified.

---

## 6. IDE / Cursor workspace notes

**`.vscode/settings.json`** (workspace) includes:

- **`editor.wordWrap`: `"on"`** — reduces horizontal scrolling in editor.
- **`files.autoSave`** after delay.
- **`files.watcherExclude`** — large subtrees (`vansh/`, `Dhairya/`, `myenv/`, …) to avoid ENOSPC **inotify** exhaustion on huge workspaces.

---

## 7. Supervisor expectations (expanded)

These come from **supervisor email + meeting-style notes** captured in project chat — treat as **direction**, not a contract on current file state.

### 7.1 Narrative & structure

- **Primary task now is paper surgery:** rewrite, tighten, and make the story **submission-shaped** (NeurIPS D&B page limits, clear contributions).
- **Introduction:** Sergei drafted motivation; student still **reads and aligns** with final framing.
- **Data curation §:** technically complete but **too long / too busy for main text** — **push depth to appendix / SI**; keep main thread readable.
- **Figure load:** many panels were generated (“dumped”); supervisor wants a **shortlist** (~order **20** candidates → **fewer** main-text figures); everything removed from main should land in a **structured appendix**, not disappear.
- **Before the main leaderboard table:** add either a **proper related-work / positioning section** or explicit **gap analysis** (what exists in literature vs what SC³ fixes; methods **not** yet benchmarked — honest gaps).

### 7.2 The “five questions” vs what we can defend

The internal **Q1–Q5** framework lives in **`paper/sections/09.ablations.tex`**. Supervisor’s assessment: **only ~three** are answered **solidly enough** for a tight camera-ready story unless scope expands again.

| Theme | Supervisor-aligned story (high level) |
|-------|----------------------------------------|
| **Transfer (Q3)** | Pretrain on **QM / CombiSolv-QM–style** solvation signal, fine-tune on SC³ — **transfer helps** (something chemistry-useful is learned); narrative + figures in **`transfer.tex`**. |
| **Scaling / “not just data quantity” (Q4)** | Experiments dumped; headline compatible with **plateau + asymptote above aleatoric reference** — story is closer to **representation / measurement quality / ceiling** than “train longer and win.” Detail in **`09a.data_scaling.tex`** + tables. |
| **Interpretability (Q2)** | **Trees + SHAP:** compare **representations** (which input featurisation wins; interpretability vs **interpretation** wording — be careful with terms). **Graph side:** what GCN attributions / substructures say about **whether chemistry is readable from the model**. Supervisor wants **3–4 tight chemistry sentences** where prose is thin — **verify with Sergei**; long-term polish goal: **Dissolvr-level** clarity on interpretation sections. |
| **Foundation models (Q5)** | Explicitly **unfinished** in §9 text — either **complete runs** or **drop / appendix** incomplete rows for submission. |

### 7.3 Experiments / numbers

- Run (or re-run) the **full benchmark sweep** on the **current split / “new” pipeline artefacts** so **`main_results.tex`** has **no permanent `---`** for methods you claim.
- **Wall-clock expectation:** order **hours** on shared GPUs; **parallelise** across **2–3 GPUs** where the harness allows.
- **HP tuning:** supervisor signal — headline metrics **may not move much** with extra HPO; if so, document protocol honestly; **do not fabricate** improvements.

### 7.4 Writing ownership

- Student owns **integration + rewrite + submission checklist**.
- Sergei / supervisor — **physics/chemistry sanity** on short interpretability blurbs and possibly intro alignment after rewrite.

---

## 8. What is done vs what is left (expanded)

### 8.1 Substantially in place (draft-complete)

- **End-to-end manuscript skeleton** in **`paper/main.tex`** — abstract through discussion; **no standalone related-work chapter yet**.
- **§3–6 technical narrative** (curation, aleatoric, benchmark, metrics) — **dense**, cited to **`DECISIONS.md`** / reports conceptually; needs **trim + notation pass**.
- **§9 ablations** as **reading guide** linking Q1–Q5 to later sections.
- **§9a data scaling** wired with equations + **`scaling_*.tex`** tables + scaling figures.
- **§Transfer** long draft + multiple figures.
- **§Interpretations** long draft + many figures (SHAP / solvent / GCN arcs).
- **Harness(es)** for training (`vansh/`, `sc3-benchmark/`) — existence confirmed; **canonical runner** for final numbers should be agreed once per group convention.

### 8.2 Incomplete or fragile (must resolve before camera-ready)

| Gap | Why it matters |
|-----|----------------|
| **`main_results.tex` holes (`---`)** | Reviewers treat missing leaderboard cells as unfinished science. |
| **ε / \(\sigma\) notation drift** | Abstract vs §4 vs §9a must **use one glossary** — otherwise calibration claims look contradictory. |
| **Related work / positioning** | Supervisor explicitly wants this **before** the big table — currently **absent** as its own section. |
| **Figure budget** | Too many panels for main text — **shortlist + appendix**. |
| **Q5 foundation models** | Text admits incomplete sweep — **finish or cut** for credibility. |
| **Q1 deferred loss ablation** | Either run a minimal version or **soften claims** that imply it was done. |

### 8.3 Polish / submission hygiene (non-science but blocking)

- NeurIPS **checklist**, **anonymisation** (double-blind), **references complete**, **appendix LaTeX wiring**, final PDF **Overfull \hbox** / figure placement pass.
- **`broad_layout.md`** aspirational numbers **vs** frozen tables — **reconcile** so abstract doesn’t overshoot evidence.

---

## 9. First actions for a new agent

1. Read **`paper/main.tex`** top-to-bottom trace + skim **`sections/`** size hotspots (`interpretations.tex`, `transfer.tex`, `03.data_curation.tex`).
2. Read **`sc3_benchmark_data_curation_v2/DECISIONS.md`** until comfortable with canonicalisation + copycat story.
3. Read **`vansh/README.md`** for runnable benchmark commands + split names.
4. Diff **`paper/tables/main_results.tex`** vs compute outputs — identify **`---`** rows.
5. Reconcile **`broad_layout.md`** aspirational claims vs **`paper/`** frozen text — flag mismatches for the student author.

---

*End of handoff summary. Update this file when milestones land (reruns complete, appendix structure committed, related work drafted).*
