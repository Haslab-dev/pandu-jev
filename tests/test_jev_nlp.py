import numpy as np
import pytest
import torch
from models.jev_nlp import (
    PanduJevNLP,
    PanduJevNLPConfig,
    build_sequence,
    clamp_temperature,
    normalized_entropy_confidence,
    render_options,
    temp_bucket,
)


@pytest.fixture(scope="module")
def jev_model():
    """Create a lightweight PanduJevNLP instance for testing."""
    config = PanduJevNLPConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,  # Fast 4-layer config for quick test execution
        num_attention_heads=4,
        vocab_size=50368,
        max_position_embeddings=512,
    )
    model = PanduJevNLP(config)
    model.eval()
    return model


def test_parameter_count_and_footprint(jev_model):
    """Verify model parameter count stays in the target lightweight regime (<25M)."""
    n_params = jev_model.num_parameters
    mem_mb = jev_model.memory_footprint_mb

    assert 10_000_000 <= n_params <= 25_000_000, f"Params out of expected bounds: {n_params}"
    assert mem_mb < 100.0, f"Memory footprint exceeds budget: {mem_mb:.2f} MB"


def test_render_options():
    """Verify option rendering across choice, score, and noul schemas."""
    # Choice with dict
    q_choice = {
        "type": "choice",
        "instructions": "Select department",
        "criteria": {"billing": "payment issues", "tech": "bug reports"},
    }
    opts = render_options(q_choice)
    assert len(opts) == 2
    assert "billing: payment issues" in opts[0]

    # Choice with list
    q_choice_list = {
        "type": "choice",
        "instructions": "Pick action",
        "criteria": ["UP", "DOWN", "LEFT", "RIGHT"],
    }
    opts_list = render_options(q_choice_list)
    assert len(opts_list) == 4
    assert opts_list[0] == "UP"

    # Score
    q_score = {
        "type": "score",
        "instructions": "Rate urgency",
        "criteria": ["low", "medium", "high"],
    }
    opts_score = render_options(q_score)
    assert len(opts_score) == 3
    assert "level 0: low" in opts_score[0]

    # Noul
    q_noul = {
        "type": "noul",
        "instructions": "Is path safe?",
    }
    opts_noul = render_options(q_noul)
    assert len(opts_noul) == 2
    assert opts_noul[0].startswith("false:")
    assert opts_noul[1].startswith("true:")


def test_sequence_builder(jev_model):
    """Verify input sequence construction and option marker index extraction."""
    state = "Agent at position (3, 4), goal at (10, 5), walls near: UP, LEFT."
    question = {
        "type": "choice",
        "instructions": "Select next movement direction",
        "criteria": ["UP", "DOWN", "LEFT", "RIGHT"],
    }

    ids, markers = build_sequence(jev_model.tokenizer, state, question, max_len=256)
    assert len(ids) > 10
    assert len(markers) == 4
    assert all(m < len(ids) for m in markers)
    assert all(ids[m] == jev_model.tokenizer.mask_token_id for m in markers)


def test_forward_pass_tensors(jev_model):
    """Verify raw tensor forward pass and output shapes."""
    batch_size = 2
    seq_len = 32
    max_options = 4

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attn_mask = torch.ones((batch_size, seq_len))
    marker_pos = torch.tensor([[4, 8, 12, 16], [5, 10, 15, 20]], dtype=torch.long)
    marker_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float)
    qtype = torch.tensor([0, 1], dtype=torch.long)

    logits, act_logits = jev_model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        marker_pos=marker_pos,
        marker_mask=marker_mask,
        qtype=qtype,
    )

    assert logits.shape == (batch_size, max_options)
    assert act_logits.shape == (batch_size, 2)
    # 4th option of sample 1 should be masked to large negative value
    assert logits[1, 3] < -1e3


def test_predict_typed_decisions(jev_model):
    """Verify full end-to-end typed decision API with zero output tokens."""
    state = {
        "customer": "Customer requests a refund for duplicate subscription invoice #1042.",
        "account_standing": "Good",
        "lifetime_value": "$1,450",
    }
    questions = {
        "dept": {
            "type": "choice",
            "instructions": "Which department handles this ticket?",
            "criteria": {
                "billing": "Invoices, refunds, and duplicate credit card charges.",
                "support": "Technical bugs, login errors, and product issues.",
                "sales": "Upgrades, enterprise contracts, and seat purchases.",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this refund ticket?",
            "criteria": ["low", "normal", "high", "critical"],
        },
        "is_fraud": {
            "type": "noul",
            "instructions": "Does this request exhibit signs of payment fraud?",
        },
    }

    result = jev_model.predict(state, questions)

    # 1. Verify response envelope
    assert result["model"] == "pandu-jev-nlp"
    assert result["parameters"] == jev_model.num_parameters
    assert "answers" in result
    assert "latency_ms" in result
    assert result["latency_ms"] > 0

    # 2. Strict zero output tokens check
    assert result["usage"]["output_tokens"] == 0
    assert result["usage"]["input_tokens"] > 0

    answers = result["answers"]
    assert len(answers) == 3

    # 3. Choice verification
    ans_dept = answers["dept"]
    assert ans_dept["type"] == "choice"
    assert ans_dept["choice"] in ["billing", "support", "sales"]
    assert len(ans_dept["probabilities"]) == 3
    assert 0.0 <= ans_dept["confidence"] <= 1.0
    assert isinstance(ans_dept["should_fallback"], bool)

    # 4. Score verification
    ans_urgency = answers["urgency"]
    assert ans_urgency["type"] == "score"
    assert 0.0 <= ans_urgency["score"] <= 3.0
    assert len(ans_urgency["probabilities"]) == 4
    assert len(ans_urgency["legend"]) == 4

    # 5. Noul verification
    ans_fraud = answers["is_fraud"]
    assert ans_fraud["type"] == "noul"
    assert 0.0 <= ans_fraud["noul"] <= 1.0
    assert isinstance(ans_fraud["verdict"], bool)
    assert 0.5 <= ans_fraud["confidence"] <= 1.0


def test_temperature_calibration_and_entropy():
    """Verify temperature scaling, bucket categorization, and normalized entropy."""
    # Temperature clamping
    assert clamp_temperature(0.1) == 0.5
    assert clamp_temperature(10.0) == 5.0
    assert clamp_temperature(1.5) == 1.5

    # Buckets
    assert temp_bucket(0, 2) == "choice:2"
    assert temp_bucket(0, 4) == "choice:3-5"
    assert temp_bucket(0, 8) == "choice:6-10"
    assert temp_bucket(0, 16) == "choice:11+"
    assert temp_bucket(2, 2) == "noul:2"

    # Normalized entropy confidence
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    conf_uniform = normalized_entropy_confidence(p_uniform, 4)
    assert abs(conf_uniform - 0.0) < 1e-4  # Zero confidence when completely uncertain

    p_certain = np.array([1.0, 0.0, 0.0, 0.0])
    conf_certain = normalized_entropy_confidence(p_certain, 4)
    assert abs(conf_certain - 1.0) < 1e-4  # 1.0 confidence when deterministic
