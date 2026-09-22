import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from graph_mvp.task_model import default_lora_targets


class Cfg:
    model_type = "glm4_moe_lite"


class FakeGlm(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Cfg()
        self.q_a_proj = nn.Linear(4, 4)
        self.q_b_proj = nn.Linear(4, 4)
        self.kv_a_proj_with_mqa = nn.Linear(4, 4)
        self.kv_b_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


def test_glm47_flash_lora_targets():
    model = FakeGlm()
    assert default_lora_targets(model) == (
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"
    )
