# QT Implementation Plan

## Finish Line

For the next month, the realistic high-quality finish line is:

- a reproducible QT pipeline for the three scoped D4RL tasks
- one reliable end-to-end replication on at least one task before broadening
- a compact ablation study
- a focused stability section around `eta`, Q-scale drift, and seed sensitivity
- a final report with both successes and failure cases

## Recommended Delivery Strategy

### Week 1: Environment + skeleton

- finalize task scope and exact benchmark settings
- set up dependencies for D4RL / Gym / MuJoCo / Maze2D
- implement dataset loading and sequence windowing with context length `K=20`
- create config-driven training and evaluation entrypoints

### Week 2: Core QT reproduction

- implement the GPT-style conditional Transformer actor
- implement the double-Q critic with target-network updates
- wire the QT training objective and inference-time action selection
- get one stable end-to-end run on `walker2d-medium-replay-v2`

### Week 3: Replication + ablations

- expand from one task to the full three-task scope
- add a policy-only Transformer baseline
- add a 1-step critic variant
- run multi-seed evaluations and clean logging

### Week 4: Stability analysis + polish

- sweep a small set of `eta` values
- track TD-error variance, Q-scale drift, gradient volatility, and collapse rate
- assemble final plots/tables
- write the final narrative: what reproduced, what did not, and why

## Priority Order

If time becomes tight, optimize in this order:

1. one trustworthy end-to-end QT result
2. reproducibility across the three chosen tasks
3. ablations
4. stability extension breadth

## Feasibility Assessment

- **Feasibility:** good if the scope stays on the three proposed tasks and the team gets stable compute early
- **Importance:** high, because the project sits at the intersection of offline RL, sequence modeling, and value-guided policy improvement
- **Hardness:** high, because the core risk is not just implementing QT, but making the critic-policy coupling stable enough to produce believable replication results

## Main Risks

- environment friction from D4RL / MuJoCo setup
- instability from critic scaling and `eta` normalization
- losing time on broad sweeps before the first solid baseline run
- unclear success criteria for what counts as a successful replication

## Questions To Settle Before Heavy Coding

- Is the success bar exact-number matching, or directional replication with clear analysis?
- Are we optimizing for strongest single-task reproduction first, or broad three-task coverage as early as possible?
- What compute is guaranteed for the team over the next month?
- How many seeds are mandatory for the final report if runtime gets tight?
- Do we prefer a clean research codebase or a fast prototype if those goals conflict?
