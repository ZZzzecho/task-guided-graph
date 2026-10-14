# Task-guided validation protocol — 2026-10-13

Authorized scope: autonomous investigation on the existing private server. Preserve
all 4563 training records (including data0), all-training final graph rewards, 876
concepts, the original eight multi-hot labels, and the existing 777-record validation.
No patient text, vocabulary contents, weights or per-record predictions go to GitHub.

Separate three questions: do graph edges help prediction; does task-guided graph
optimization improve on fixed graphs and matched search; does GRPO itself add value?
Report unsupported outcomes as well as improvements. A negative bounded experiment
does not prove that the entire task-guided research direction is impossible.

Stage 1 reuses the existing initial MNGM graph and embeddings. Train text-only,
concept-prefix without edges, and fixed-graph models for paired seeds 11/17/23,
one complete epoch each, identical train order/LoRA initialization/head initialization,
batch size 4, learning rate 1e-4, maximum text length 512 and threshold .5. Preserve
legacy activation in this first comparison to isolate topology. Report macro-AP
(primary), macro/micro-F1, BCE, per-label metrics and graph activation aggregates.
Bootstrap intervals on the reused validation are descriptive and are not test evidence.

In parallel, improve ADMM warm starts while retaining SPD/KKT precision. Benchmark
on the actual stored covariance/penalty problem before expanding graph search.
Audit data0 provenance and stratified distributions; do not discard training rows.

Subsequent stages will diagnose/repair activation, compare fixed and task-guided
graphs against matched simple searches, then test transfer across initial graphs.
They require recorded code/configuration/budget choices before running. Material
algorithm changes are separate commits. Do not silently replace MNGM with a graph
reader and claim equivalent statistical guarantees.

Final confirmation: lock method, training/threshold settings and comparators before
opening the test split. Primary target is mean macro-AP improvement at least .005
over the strongest matched comparator, with no material macro-F1 loss (> .002),
paired directional support and seed consistency. These are research decision targets,
not clinical standards. Current validation has already been used for exploration.
Patient IDs are unavailable, so record independence remains a limitation.

Pilot amendment, 2026-10-13: the first seed yielded macro-AP .174/.206/.190
for text/concept/graph, far below the earlier .865 fixed-graph checkpoint.
Stop expansion of that pilot and diagnose reproduction before drawing topology
conclusions. Its dropout RNG was reset to the seed after initialization; record and
restore the post-initialization RNG for subsequent paired arms to match the original
training path. Separately expose FP32 trainable heads and mean pooling as recorded
experimental options; neither is yet established as the cause or remedy.
Frozen-representation probes fit FP32 linear heads on all training records, compare
last/mean pooling and uniform/sqrt-positive weighting, and leave test unopened.
These probes diagnose representations and optimization; they are not graph-policy
evidence. Conditional penalty grids hold the representation precision B fixed and
must be identified as conditional Glasso rather than a full alternating MNGM refit.

Conditional-grid evaluation uses the original frozen task checkpoint. Select the
fitted graph with highest macro-AP on all 4563 training records, subject to fixed-B
unpenalized NLL increase per concept <= .01 and edge count <= twice the initial
count at |partial correlation| > 1e-5. Record these exploratory gates before scores
are computed. Validation evaluates that training-selected candidate; zero and
permuted graphs are sensitivity controls and cannot enter selection. No GRPO
advantage or full alternating-MNGM claim follows from this conditional experiment.

Solver-only follow-up: compare adaptive residual balancing (ratio 10, factor 2,
every 25 steps through iteration 1000, scaled-dual rescaling, then fixed rho) and
fixed rho 10/100 against the existing +5% penalty benchmark. Preserve all stopping
tolerances and compare objective/precision before adoption. The optional setting
defaults off. See Boyd et al. (2011), section 3.4.1:
https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf.

Full-population contraction benchmark: an optional Torch device backend retains
float64 transformed samples, uses the same matrix contractions and CPU ridge/SPD
projection, and leaves NumPy as default. Compare both covariances and unpenalized
likelihood at the actual 4563 x 64 x 876 shape before adoption. This benchmark
alone cannot establish a speedup for the entire alternating fit.

Stabilized paired controls: seeds 11/17/23, all three arms, two full epochs,
FP32 classifier and graph modules, mean pooling, post-initialization paired RNG,
otherwise the same LoRA, lr=1e-4, batch=4, full train and validation populations.
Run seed 11 first; if any arm macro-AP is below .5, preserve/report that diagnostic
failure and halt seed expansion to inspect training before spending further budget.
This is an exploration health rule, not a rule for selecting successful seeds in
a final comparison. Successful controls still require independent confirmation.

Prepare a second task-guided optimizer baseline: active-set implicit weighted
Glasso derivatives. The local derivative retains free diagonals and uses the
upper-triangle penalty convention. Numerical adjoint convergence and a fresh
SPD/KKT-certified solve are required before using a direction. This is gradient
optimization rather than GRPO and supplies no confidence intervals or bootstrap
stability evidence. Candidate/selection settings must be recorded before launch.
General background on differentiating through convex programs: Agrawal et al.
(2019), https://papers.neurips.cc/paper_files/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html.
The active-set Glasso adjoint here is a project-specific derivation, checked by
finite differences; it is not a reproduction of that paper's software.

Directed optimizer experiment: freeze each stabilized graph checkpoint. Compute
the macro pairwise logistic-ranking surrogate gradient on all 4563 records
(temperature 1, plus .1 uniform BCE); its surrogate is not AP. Map the precision
gradient through the certified active-set adjoint (CG residual <=1e-6, max1000).
Eligible penalty coordinates have initial |partial correlation| >=.001; empirical
fit support is not confidence. Scale log-penalty descent by the 95th percentile,
clip directions to [-1,1], and try steps [.1,.3,.7] with penalties clipped [.05,3].
Compare three matched random candidates from one seed+1001 permutation of exactly
the same eligible edge values, and three uniform candidates [.2,.4,1.2]. Each family
includes the unchanged baseline. A fresh strict Glasso solve uses rho10; one
adaptive-rho1 retry is allowed for numerical failure, with unchanged tolerances.
Conditional NLL increase per concept <=.01 and edges <=2xinitial remain the gates;
full-training macro-AP alone selects each family's candidate. Refit each selected
penalty with full alternating patient-MNGM, rho10 and the verified float64 GPU
contractions, then recheck full NLL/complexity and training AP; reject to the initial
graph when these fail. No validation selection or test access. Compare final
methods with the strongest matched text/concept/fixed-graph controls. This is a
bounded gradient-versus-search test, not a GRPO advantage or global optimum claim.
### Frozen-feature readout diagnosis, 2026-10-13

After the stabilized seed-11 controls recovered graph/concept training but the
text arm remained unhealthy, do not expand or filter those failed controls into
a multi-seed efficacy claim. Independently fit fresh FP32 linear heads to the
already cached mean text features of both the original trained backbone and the
unadapted base backbone. Use all 4563 training records, seeds 11/17/23, 300 Adam
steps at .01, training-only feature normalization, and the same .001 mean-squared
weight penalty. Report uniform BCE (primary) and square-root positive weighting
(secondary), every seed, and training metrics for the original/data0 sources.
No selection by validation and no test access. The original frozen backbone was
trained with graph, so its readouts do not establish independent text-only
training performance or graph efficacy. This check isolates readout stability
and the attainable performance of the cached representations.


### Cached mention graph reader pilot, 2026-10-13

Predeclare a cheap architectural diagnostic before task-guided search on this
reader. Use the same original frozen text features and the corresponding frozen
uniform-BCE text head for seeds 11/17/23. Binary literal keyword mentions gate
nodes. Messages are a_i sum_j rho_ji a_j, scaled by per-node standard deviation
of the initial graph messages on TRAIN only, floored at .1 and fixed across arms.
Train a residual tanh node reader (learned self/message scales, bias and an
8-label linear readout initialized to zero), 300 full-training Adam steps at
.01 with .001 mean-squared readout weight penalty. Compare text, concept/no-edge,
initial graph and a single fixed node-permuted graph. Same residual capacity and
initialization in the last three arms; frozen text head throughout. Report every
arm and seed, no validation selection, test closed. This changes the receiver and
is exploratory graph mechanism evidence on one frozen backbone, not a GRPO,
statistical-confidence or end-to-end independent text training claim. Literal
mentions can include negated findings and are not annotated medical entities.

The unregularized cached-reader pilot improved training AP to approximately .98
but reduced validation AP from .901 to .839 (concept) or .845 (graph). The graph
advantage over concept/permuted does not clear the stronger text reference.
Follow up with predeclared residual-logit penalties [.1,1,10], leaving all other
settings, all seeds and all training records unchanged. Penalize the mean square
of the correction to the frozen text logits; this checks whether overfitting the
new residual is reversible. Report the complete regularization panel; do not
select the validation winner and treat it as confirmed efficacy. Graph search
and the statistical-evidence scheme remain separate questions.

### Receiver aligned with the statistical variables

The mention-reader panel did not meet the AP/F1 target. Before rejecting the
research direction, test a receiver that uses the same patient-concept H as the
MNGM fit. Build 777 validation matrices with the exact saved training projection,
concept order, frozen encoder, joint-evidence template, top-2/chunk64/max512
settings; batching is increased to 32 for efficiency. Validation matrices are
never included in the statistical fit. Assert the saved projection identity and
mark this cache source_split=validation after completion. No test access.

Gaussianize each H cell using the TRAIN empirical CDF, matching the training
MNGM rank transform including ties. Use a shared trainable 64-to-8 task projection,
concept identity vectors, signed absolute-degree-normalized graph message and
8-label attention pooling. This projection changes only the receiver; the
statistical basis is fixed. A zero-initialized residual label layer preserves the
same seed's frozen cached uniform-BCE text head at initialization. All three
residual arms share architecture, initialization, training order and capacity.
Compare no-edge concept, initial graph and fixed node permutation; include the
frozen text reference. Seeds 11/17/23, 5 full-training epochs, batch128, Adam .003,
logit-correction penalty 1 and .001 mean-squared label-weight penalty. Report every
arm, no validation selection. These are receiver seeds on one frozen backbone,
not independent end-to-end training or a GRPO/uncertainty result. GPU is shared
with other frozen-model experiments, so elapsed times are not isolated benchmarks.

### Frozen graph intervention audit

User explicitly asks to verify actual graph usefulness, including why learned
and random graphs appear equally ineffective. Audit all three completed mention
reader regularization settings [.1,1,10] without retraining. For each setting and
reader seed11/17/23, freeze both the true-graph-trained and permutation-1001-trained
teachers; intervene with true graph, zero graph and permutations1001..1005.
Assert native predictions reproduce saved private scores before comparing.
Report every intervention. Any drop after a swap establishes graph dependence,
not superior graph quality by itself; random-trained teachers control for generic
mismatch between the trained representation and a new graph. Matched retraining
controls, seed variation and untouched-test confirmation remain necessary for
positive efficacy claims. No validation selection or test access.

### Held-out statistical graph information, 2026-10-14

After the aligned receiver showed almost no diagnostic difference, independently
evaluate whether the learned graph contains statistical information. Use all777
validation H matrices, the completed training-ECDF rank caches and training-only
cell means. Hold the initial MNGM row precision B fixed. Compare initial concept
precision, a no-edge diagonal fit using training covariance at the same B, and
five node permutations of normalized precision, rescaled to preserve ORIGINAL
precision diagonal. These nulls preserve partial edge weights/support size and
SPD; their normalized precision spectrum and log determinant match the original.
Compute leave-one-concept-out reconstruction from every other concept in the
same record, joint Gaussian NLL/cell, and conditional pseudo-NLL/cell. Predicting
a concept must exclude its own value. Use 1000 paired record bootstrap resamples
only as descriptive intervals; patient IDs are unavailable. Do not use labels,
refit with validation, select a null, or open test. A positive reconstruction or
density result establishes statistical representation information, not improved
diagnosis or GRPO effectiveness. Verify densities against a full Kronecker-normal
oracle and conditional reconstruction against explicit Gaussian conditional means.
