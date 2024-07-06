# Phase 3 — Progress & Updates

**Status:** COMPLETE

---

## Task Checklist

- [x] Solute-solvent-source grouping (10,877 pairs, 11,785 triples)
- [x] Copycat/duplication detection (14 exact, 125 near-duplicates)
- [x] Apelblat/van't Hoff curve fitting (11,239 Apelblat, 20 van't Hoff, 0 failures)
- [x] True inter-lab variability computation (610 pairs compared, median MAE 0.055)
- [x] Source reliability ranking (369 DOIs ranked, 285 Hall of Fame, 30 Hall of Shame)
- [x] Stratified aleatoric limit analysis (median 0.055 overall, P90 = 0.475)
- [x] Phase report with figures (7 figures)

## Session Log

| Date | Agent | What was done | Outcome |
|------|-------|---------------|---------|
| 2026-04-14 | Main | Wrote source_analysis.py (Steps 3.1-3.6 + plotting) | Complete script |
| 2026-04-14 | Main | Ran full analysis pipeline | All steps succeeded |
| 2026-04-14 | Main | Wrote Phase 3 report | phase_03_source_analysis.md |

## Notes & Findings

- **Copycat problem is real but limited:** 18.9% of source pair comparisons are suspected duplicates (exact or near-duplicate values). Excluded from inter-lab calculations.
- **Inter-lab variability much lower than literature:** Median MAE = 0.055, far below the commonly cited 0.6–0.8. The literature number conflates quality tiers and includes copycats.
- **Apelblat fits are excellent:** Mean R² = 0.9948, 98.4% have R² ≥ 0.95. Zero failures.
- **30 DOIs in Hall of Shame** (MAE ≥ 0.6). Worst: 10.1021/je4000718 with MAE = 4.35.
- **Only 3 pairs have ≥5 sources** — easy tier has insufficient data for robust statistics.
- **For Phase 4:** Use P90 (0.475) as consensus window threshold, remove Hall of Shame DOIs.
