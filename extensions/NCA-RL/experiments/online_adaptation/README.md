# Tier 2 — Necessity-guided O2O adaptation

**Status: future work** (paper §5.3, scoped beyond the workshop submission).

Train NCA-T end-to-end on `antmaze-medium-diverse-v2`, run online adaptation for $B_{\text{on}} = 10^5$ steps, and compare against:

- (a) uniform online finetune (Cal-QL or IQL) — isolates necessity contribution
- (b) uncertainty-only gating using twin-Q disagreement $\mathcal{U}(s)$ — isolates disagreement signal
- (c) Q-credit-weighted finetune — isolates sufficiency baseline
- (d) direct $\mathcal{N}^{\text{off}}$ gating without learned head — isolates head contribution
- (e) no online calibration — isolates drop-test data value

Estimated cost: 2–3 GPU-weeks. Out of scope for the May 7 submission.
