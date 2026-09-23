"""Task-guided Bootstrap-MNGM graph learning. Test data never enters graph search."""

__version__ = "0.4.0"

from .types import (GraphSnapshot, GraphState, GraphCandidate, ActionRecord,
                    ActionGroup, GroupPolicyExperience)
from .environment import GraphEnvironment, CandidateBuilder, concept_axis_kkt_frontier
from .weighted_glasso import WeightedGraphicalLasso
from .estimators import (VectorGGMEstimator, MNGMEstimator, VECTOR_MODE,
                         PATIENT_MATRIX_MODE, BOOTSTRAP_MATRIX_MODE)
from .policy import GRPOPolicy
from .runner import GraphPhaseRunner
from .graph_tokens import SoftGraphTokenizer, bootstrap_concept_prototypes
from .patient_repr import (PatientConceptMatrixBuilder, PatientMatrixDataset,
                           ConceptVocabulary, Qwen3EmbeddingEncoder)
from .task_prototypes import task_lm_concept_prototypes, build_task_soft_graph_tokenizer
from .task_model import attach_task_lora, load_local_causal_lm

__all__ = [
    "GraphSnapshot", "GraphState", "GraphCandidate", "ActionRecord", "ActionGroup",
    "GroupPolicyExperience", "GraphEnvironment", "CandidateBuilder",
    "concept_axis_kkt_frontier", "WeightedGraphicalLasso", "VectorGGMEstimator",
    "MNGMEstimator", "VECTOR_MODE", "PATIENT_MATRIX_MODE", "BOOTSTRAP_MATRIX_MODE",
    "GRPOPolicy", "GraphPhaseRunner", "SoftGraphTokenizer",
    "bootstrap_concept_prototypes", "PatientConceptMatrixBuilder",
    "PatientMatrixDataset", "ConceptVocabulary", "Qwen3EmbeddingEncoder",
    "task_lm_concept_prototypes", "build_task_soft_graph_tokenizer",
    "attach_task_lora", "load_local_causal_lm", "__version__",
]
