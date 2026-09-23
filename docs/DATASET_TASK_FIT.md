# Dataset–Task Fit Principles for Graph-RL

> Status: locked research principle.  
> Purpose: define what kinds of datasets can validly test whether Graph-RL improves a concept graph, rather than merely injecting informative concept identities.

## 1. Core requirement

The downstream target must **not** be equivalent to concept presence, and should not be strongly predictable from a single activated concept.

For sample (d), the current graph injection pipeline is

[
x_d
ightarrow
a_d
ightarrow
W_d=D(a_d) W D(a_d)
ightarrow
	ext{graph tokens}
ightarrow
	ext{LLM / downstream task}.
]

Here (a_d) is the sample-specific concept activation vector and (W) is the global learned concept graph.

Because (a_d) already reveals which concepts are relevant to the current sample, a downstream task such as “predict the active concepts” is structurally confounded: the gain may come from node identity / node activation rather than from the learned edge structure.

What we want to identify is

[
Delta_{	ext{edge structure}},
]

not

[
Delta_{	ext{node identity}}+Delta_{	ext{edge structure}}.
]

Therefore the primary downstream target should require **composition, interaction, or reasoning across multiple concepts**.

## 2. Preferred dataset properties

A strong dataset for this project should have:

1. Natural-language text as the sample input.
2. A fixed, shared domain concept vocabulary across samples, preferably hundreds to low-thousands of concepts.
3. Concepts that are repeatedly reused across many samples in the same semantic domain.
4. A meaningful external concept structure when available (ontology, hierarchy, thesaurus, relation graph, etc.), used only for evaluation unless explicitly stated otherwise.
5. A downstream target that is different from the concept set itself.
6. A downstream task whose loss plausibly depends on concept interactions rather than one dominant concept.
7. Enough samples to estimate a stable global concept graph.
8. A design that lets the same sample activation (a_d) be reused while only changing the graph (W).

## 3. Bad primary-task pattern

Avoid using concept prediction itself as the main Graph-RL reward when those same concepts define the graph nodes.

Example:

[
	ext{text}
ightarrow
a_d
ightarrow
W_d
ightarrow
	ext{predict active concept labels}.
]

This makes it difficult to distinguish whether a performance gain comes from a better graph or simply from exposing highly informative activated nodes.

Such tasks may still be used as auxiliary analyses, but should not carry the main causal claim about graph quality.

## 4. Better downstream-task pattern

Prefer targets that are downstream consequences of multiple concepts and their relations.

Examples:

- diagnosis / outcome prediction from symptoms, findings, anatomy, tests, and history;
- citation or prior-art retrieval from a scientific or patent document;
- entity / document retrieval where the answer cannot be identified by a single concept;
- multi-step generation or reasoning tasks in which multiple concepts must be composed.

The ideal structure is:

[
	ext{many local signals}
ightarrow
	ext{concept interaction structure}
ightarrow
	ext{target distinct from concept identity}.
]

This is why the original sports-medicine diagnosis setting is a strong match: activated symptoms, findings, anatomical regions, and tests are informative, but the final diagnosis depends on their combination.

## 5. Required graph-structure ablations

When evaluating graph injection, keep sample activation fixed and vary only the edge structure:

[
W_{	ext{learned}},
quad
W_{	ext{identity / no-edge}},
quad
W_{	ext{shuffled}},
quad
W_{	ext{frequency / heuristic}},
quad
W_{	ext{external ontology}}
]

when applicable.

All variants must use the same sample (x_d), the same concept vocabulary, and the same activation (a_d).

The comparison of interest is therefore closer to:

[
	ext{Task}(D(a_d)W_{	ext{learned}}D(a_d))
-
	ext{Task}(D(a_d)W_{	ext{control}}D(a_d)),
]

which isolates the value of edge structure much more cleanly.

## 6. Current dataset-selection consequence

Dataset selection should prioritize domains where:

- concept nodes are dense and reusable rather than sample-specific;
- a single domain can naturally support (Papprox 500)–(2000) shared concepts;
- the downstream task is not simply ontology/concept classification;
- an external graph/hierarchy exists for independent structural evaluation when possible.

Current high-priority directions are therefore domain-specific scientific literature or patent corpora with structured concept systems plus citation / prior-art style downstream tasks.
