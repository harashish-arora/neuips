# Agent Onboarding — Read This First

## Before You Do Anything Else

1. **Read `STATUS.md`** — current phase and overall state
2. **Read `HANDOFF.md`** — check the handoff log at the bottom
3. **Read the latest handoff note** in `handoffs/` (if any exist)
4. **Read the plan and progress files** for the current phase (e.g., `phase_01_data_acquisition/plan.md` and `progress.md`)

Only then should you begin work.

---

## Repo Layout

```
Molmerger_Anon/
├── Plan/                          # <-- YOU ARE HERE — planning & coordination
│   ├── AGENT_START_HERE.md        # This file
│   ├── STATUS.md                  # Master status tracker
│   ├── HANDOFF.md                 # Handoff protocol & log
│   ├── handoffs/                  # Individual handoff notes
│   ├── phase_01_data_acquisition/ # Plan + progress for Phase 1
│   ├── phase_02_merging_cleaning/ # Plan + progress for Phase 2
│   ├── ...                        # (one folder per phase, 1-9)
│   └── phase_09_paper_writing/
│
├── Dissolvr/                      # EXISTING CODEBASE — datasets, baselines, models
│   ├── regime-i/                  # Single-solvent (aqueous) models & data
│   ├── regime-ii/                 # Multi-solvent models & data
│   ├── baselines/                 # Baseline implementations
│   ├── apelblat/                  # Apelblat curve fitting
│   ├── explainer/                 # Model explainability
│   └── ...
│
├── sc3-benchmark/                 # NEW PROJECT — the SC3 benchmark workspace
│   ├── data/raw/                  # Original datasets
│   ├── data/intermediate/         # Intermediate cleaning stages
│   ├── data/clean/                # Final train/val data
│   ├── data/sc3/                  # SC3 challenge test sets
│   ├── src/                       # All code (data, analysis, metrics, models, etc.)
│   ├── reports/                   # Phase completion reports
│   ├── pass-ons/                  # Legacy handoff location (use Plan/handoffs/ instead)
│   ├── figures/                   # Generated figures
│   ├── tables/                    # Generated tables
│   ├── configs/                   # Hyperparameter configs
│   └── paper/                     # LaTeX paper
│
└── sc3_agent_protocol.md          # Full protocol document (the "bible")
```

## Key Conventions

- **Code goes in `sc3-benchmark/src/`** — modular `.py` scripts with argparse, no notebooks
- **Reports go in `sc3-benchmark/reports/`** — one per phase
- **Figures go in `sc3-benchmark/figures/`** — .png (300 DPI) + .pdf
- **Planning lives in `Plan/`** — update progress.md as you complete sub-tasks
- **Handoffs live in `Plan/handoffs/`** — write one before ending any session

## Rules

1. **Sequential phases** — don't skip ahead
2. **Honesty over results** — report what you find, not what you wish
3. **CPU only** — no GPU access
4. **Update progress as you go** — check boxes in the relevant progress.md
5. **Write a handoff note when ending** — even if you finished cleanly
6. **Reuse Dissolvr code** — don't reinvent what already exists

## Quick Reference: Phase Dependencies

```
Phase 1 (Data) → Phase 2 (Merge/Clean) → Phase 3 (Source Analysis) → Phase 4 (Dataset Construction)
                                                                          ↓
Phase 5 (Metrics) ←── needs Phase 4 dataset
Phase 6 (Methods) ←── needs Phase 4 dataset for test runs
                                                                          ↓
Phase 7 (Benchmarking) ←── needs Phases 4, 5, 6
Phase 8 (Interpretability) ←── needs Phase 7 trained models
Phase 9 (Paper) ←── needs all prior phases
```
