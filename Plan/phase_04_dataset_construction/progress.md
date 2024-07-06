# Phase 4 — Progress & Updates

**Status:** COMPLETE

---

## Task Checklist

- [x] Deep aleatoric limit analysis (ε_aleatoric = 0.032 log S composite)
- [x] Per-solvent, per-temperature, per-LogS stratification
- [x] Bootstrap CIs and distribution fitting
- [x] Training dataset constructed (80,487 rows, NO aleatoric removal)
- [x] Validation dataset constructed (14,139 rows, 14.9%)
- [x] SC3-Easy constructed (503 points, 23 solutes, consensus ground truth)
- [x] SC3-Medium constructed (202 points, 9 solutes, moderate characterization)
- [x] SC3-Hard constructed (120 points, 5 solutes, OOD challenge)
- [x] Zero-leakage verified (all 6 checks CLEAN)
- [x] Statistics report and data quality certificate written

## Session Log

| Date | Agent | What was done | Outcome |
|------|-------|---------------|---------|
| 2026-04-14 | Main | Wrote aleatoric_deep.py — proper theory analysis | 7 figures, composite ε = 0.032 |
| 2026-04-14 | Main | Wrote build_sc3.py — dataset construction | All 5 datasets built |
| 2026-04-14 | Main | Ran both scripts, verified outputs | Zero leakage confirmed |
| 2026-04-14 | Main | Wrote Phase 4 report | phase_04_dataset_construction.md |

## Notes & Findings

- **Composite aleatoric limit = 0.032 log S** (median) — dramatically lower than literature's 0.6–0.8
- Literature's 0.6–0.8 corresponds to our P95 (0.634) — they characterized the tail, not the typical case
- Error distribution is log-normal (heavy right tail)
- DMSO has lowest measurement uncertainty (0.012 median MAE); acetonitrile highest (0.035)
- Training set has NO aleatoric removal — models must be robust to real-world noise
- SC3-Hard uses OOD framing (novel solutes not in training)
- 37 test solutes removed from train/val at molecule level
