# Extensions

Vendored copies of the two follow-up codebases referenced in [`../report/report.pdf`](../report/report.pdf). Each is a standalone project with its own README; they are bundled here so a grader landing on this repo has the full picture in one place.

| Folder | What it is | Status (per the final report) |
|---|---|---|
| [`NCA-RL/`](NCA-RL/) | *Necessity-aware Credit Assignment* — a learnable counterfactual-necessity score (Q-margin proxy + ensemble MLP head) on top of a frozen offline IQL critic, validated by state-restored counterfactual drop tests. | Tier-0 (gridworld) done; Tier-1 (`maze2d-large-dense-v2`) headline finding: `ρ(Qϕ, N_off) = −0.033` — necessity is empirically orthogonal to Q-value. |
| [`SARD-QT/`](SARD-QT/) | *QSC-QT / SARD-QT* — research-extension repo that initially explored a supervised reverse-credit critic fused with QT (the failed extension documented in the report) and now studies Q-spread collapse with spectral candidate generation. | Failed-extension diagnostics retained for reproducibility; active line of work has pivoted to QSC-QT. |

These are **vendored snapshots** (history dropped): they are not git submodules. To pull upstream updates, re-clone and replace the folder.

Upstream sources:
- `NCA-RL` → https://github.com/BobbyZbp/NCA-RL
- `SARD-QT` → https://github.com/BobbyZbp/SARD-QT
