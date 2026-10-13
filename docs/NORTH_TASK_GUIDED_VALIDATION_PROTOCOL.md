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
