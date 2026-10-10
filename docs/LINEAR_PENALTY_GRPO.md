# Linear penalty GRPO trial

`--policy grpo-linear` adds an opt-in alternative to the unchanged default
32-pair GRPO. It controls all P(P-1)/2 concept-pair penalties through eight
shared coefficients. It does not directly overwrite the precision graph.

Features, in order: intercept; fixed Qwen concept-prototype cosine; absolute
effective-covariance correlation; absolute partial correlation; active-edge
indicator; solver-derived KKT frontier; mean degree divided by P-1; training
reward-panel task relevance normalized by its maximum absolute value.

Rows are divided by their L1 norm (at least one). The policy encodes the mean
and standard deviation of these eight features with a 16 -> 32 -> 32 -> 24 MLP.
Eight categorical heads choose coefficients from {-1, 0, +1}. The coefficients
are discrete in this first version; the resulting pairwise changes are continuous:

```
delta_ij = phi_ij @ coefficients
log(lambda'_ij) = clip(log(lambda_ij) + eta * delta_ij)
```

The existing eta=log(1.1), [0.01,4] penalty bounds, symmetry and zero diagonal
are preserved. Floating-point endpoint overshoot is clipped. Coefficient
log-probabilities, exact KL to the fixed initial policy and entropy are summed
over eight heads, never over the expanded pairs. The existing clipped GRPO
update, invalid-candidate masks, reward normalization and independent validation
acceptance are reused. This policy has no edge reward memory. Features and
expanded actions are frozen per sampled group for on-policy updates.

`LinearCandidateBuilder` includes every pair, including nonactive pairs outside
the old frontier cap. Prototype metadata must match the complete concept order.
Full alternating MNGM is still solved per candidate with warm Theta and B;
numerical failures are masked with no graph acceptance. The original batch
policy remains available through `--policy grpo`.

## Full-data exploratory trial

`scripts/run_linear_penalty_trial.sh` uses all 2048 existing long-text matrices,
800 concepts and 64 representation coordinates, without rebuilding H. It verifies
and reuses the completed fitted graph/B: cache file checksums, original config
hash, effective solver/MNGM settings, sample fingerprint, concept order,
recomputed effective covariance, positive definiteness and original KKT tolerance.
The inner budget is explicitly 10000; original tolerances remain unchanged.

The trial warms task LoRA for 50 steps on the 1024-query training pool, then
freezes it for 2 phases x 2 GRPO groups x 4 candidates = 16 full MNGM candidates.
Reward/validation panels remain 128/256 queries. `--adapt-steps 0` explicitly
disables post-acceptance task adaptation. Test pools are not supplied or read.
This isolates immediate graph changes within the trial; it is not yet a matched
multi-seed comparison proving linear GRPO outperforms the old policy.

Run in the existing server graph_rl environment:

The H20 trial explicitly loads the BF16 task model on `cuda:0`; its checkpoint
is about 62.4 GB and the server has about 140 GiB of GPU memory. The fixed encoder
snapshot requires all model weights/buffers to be materialized. Automatic CPU/disk
offload can leave meta tensors and is not supported by this snapshot implementation.
Startup failures now mark an initialized run manifest as failed and preserve
the original exception. The initial 2026-10-09 trial failed during snapshot creation
before warmup or any GRPO candidate; retain that output and restart in a new directory.

```bash
source /laijizheng/miniconda3/etc/profile.d/conda.sh
conda activate graph_rl
cd /laijizheng/task_guided_graph_mvp_v0.3.0_2026-09-22
mkdir -p logs
OUTPUT_DIR=outputs/grpo_linear_trial_$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR
nohup bash scripts/run_linear_penalty_trial.sh > "$OUTPUT_DIR.launch.log" 2>&1 < /dev/null &
echo $! > "$OUTPUT_DIR.pid"
```

`progress.jsonl` records solver and policy progress. Candidate/phase reports
record coefficients, joint probability, reward, validity, actual changed penalties,
Rho change, exact policy KL/entropy and acceptance. Phase checkpoints retain policy
and graph proposals. Full candidate solves can be slow; estimate time after
measuring the first completed solves, not from policy-network runtime.

Validation: 46 focused server tests passed before the trial, including coefficient
probability replay, parameter updates, invalid/flat groups, bound/symmetry/keep
behavior, original policy/environment tests, real solver/reward/validation flow,
verified graph reuse and corruption rejection, and retrieval/MNGM integration.
