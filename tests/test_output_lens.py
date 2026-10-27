"""Output lenses (spd/models/output_lens.py): the target losses of the stimulus and residual lenses
read what they claim to read.

- Prompt batches with a stimulus keep the prompt's own tokens, decompose only prompt positions and
  record where the stimulus ends; trimming keeps the stimulus.
- Under the stimulus lens, the masked-vs-original KL at the decomposed position equals the KL of the
  next-token logits at the stimulus end, with components substituted only at the decomposed
  position (the stimulus runs on the original weights).
- The residual lens returns Hugging Face's `hidden_states` after the chosen layer, divided per
  channel by the given standard deviations.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from spd.configs import GlobalCiConfig, LastKPositions, StimulusLens, TokenPositions
from spd.experiments.lm.prompts_dataset import load_prompts_dataset, trim_to_loss_positions
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos, route_only_selected_positions
from spd.models.output_lens import ResidualOutputLens, StimulusOutputLens
from spd.utils.general_utils import (
    POSITION_MASK_KEY,
    STIMULUS_END_KEY,
    calc_sum_recon_loss_lm_at_positions,
)
from spd.utils.module_utils import ModulePathInfo

TINY_QWEN = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
PROMPTS = [
    "The city of Krasnodar is in Russia.",
    "Paris is in France.",
    "The Spanish word 'uno' means 'one'.",
]
STIMULUS = " Is this true or false? Answer:"
LENS = StimulusLens(type="stimulus", stimulus=STIMULUS)
MAX_SEQ_LEN = 32


def _tiny_model(decomposed_layers: tuple[int, ...]) -> ComponentModel:
    target = AutoModelForCausalLM.from_pretrained(TINY_QWEN, dtype=torch.float32).eval()
    target.requires_grad_(False)
    # The random tiny model's logits have std ~0.01, so its next-token distributions are nearly
    # uniform and every KL is at float-noise level (~1e-7); scaling the final norm by 1000 gives
    # logits of std ~10 and KLs of order 0.01-1, large enough to tell positions apart.
    with torch.no_grad():
        target.get_parameter("model.norm.weight").mul_(1000.0)
    torch.manual_seed(0)
    return ComponentModel(
        target_model=target,
        module_path_info=[
            ModulePathInfo(module_path=f"model.layers.{i}.mlp.down_proj", C=4)
            for i in decomposed_layers
        ],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        sigmoid_type="leaky_hard",
        pretrained_model_output_attr="logits",
    )


def test_stimulus_batches_keep_prompt_tokens_and_mark_the_stimulus_end() -> None:
    tok = AutoTokenizer.from_pretrained(TINY_QWEN)
    stim_ids = tok.encode(STIMULUS, add_special_tokens=False)
    for loss_positions in (LastKPositions(type="last_k", k=1), TokenPositions(type="tokens")):
        batch = load_prompts_dataset(PROMPTS, tok, MAX_SEQ_LEN, loss_positions, LENS)[:]
        for row, prompt in enumerate(PROMPTS):
            prompt_ids = tok(prompt)["input_ids"]
            n = len(prompt_ids) + len(stim_ids)
            assert batch["input_ids"][row, :n].tolist() == prompt_ids + stim_ids
            assert int(batch[STIMULUS_END_KEY][row]) == n - 1
            selected = batch[POSITION_MASK_KEY][row].nonzero().flatten().tolist()
            expected = (
                [len(prompt_ids) - 1]
                if isinstance(loss_positions, LastKPositions)
                else list(range(len(prompt_ids)))
            )
            assert selected == expected, (loss_positions, selected)
        trimmed = trim_to_loss_positions(batch)
        assert trimmed["input_ids"].shape[1] == int(batch[STIMULUS_END_KEY].max()) + 1
        assert torch.equal(trimmed[STIMULUS_END_KEY], batch[STIMULUS_END_KEY])


def test_decompose_stimulus_adds_exactly_the_stimulus_tokens() -> None:
    tok = AutoTokenizer.from_pretrained(TINY_QWEN)
    n_stim = len(tok.encode(STIMULUS, add_special_tokens=False))
    lens = StimulusLens(type="stimulus", stimulus=STIMULUS, decompose_stimulus=True)
    batch = load_prompts_dataset(
        PROMPTS, tok, MAX_SEQ_LEN, LastKPositions(type="last_k", k=1), lens
    )[:]
    for row, prompt in enumerate(PROMPTS):
        n_prompt = len(tok(prompt)["input_ids"])
        selected = batch[POSITION_MASK_KEY][row].nonzero().flatten().tolist()
        # the prompt's last token (the period) and every stimulus token, nothing else
        assert selected == list(range(n_prompt - 1, n_prompt + n_stim)), selected
        assert int(batch[STIMULUS_END_KEY][row]) == n_prompt + n_stim - 1


def test_stimulus_lens_reads_the_stimulus_end() -> None:
    tok = AutoTokenizer.from_pretrained(TINY_QWEN)
    model = _tiny_model(decomposed_layers=(0, 1))
    batch = trim_to_loss_positions(
        load_prompts_dataset(PROMPTS, tok, MAX_SEQ_LEN, LastKPositions(type="last_k", k=1), LENS)[:]
    )
    ids, decomposed, end = batch["input_ids"], batch[POSITION_MASK_KEY], batch[STIMULUS_END_KEY]
    # All components off and no delta: the decomposed MLPs output zero at the decomposed position.
    masks = {name: torch.zeros(*ids.shape, 4) for name in model.target_module_paths}
    mask_infos = route_only_selected_positions(make_mask_infos(masks), decomposed)

    with torch.no_grad(), model.output_lens(StimulusOutputLens(end)):
        target = model(ids)
        masked = model(ids, mask_infos=mask_infos)
    lens_loss, n = calc_sum_recon_loss_lm_at_positions(masked, target, "kl", decomposed)

    with torch.no_grad():
        target_logits = model(ids)
        masked_logits = model(ids, mask_infos=mask_infos)
    rows = torch.arange(len(PROMPTS))

    def kl(at: torch.Tensor) -> torch.Tensor:
        log_q = torch.log_softmax(masked_logits[rows, at], dim=-1)
        p = torch.softmax(target_logits[rows, at], dim=-1)
        return F.kl_div(log_q, p, reduction="sum")

    assert n == len(PROMPTS)
    assert kl(end) > 1e-2, "KL too small to test anything"
    torch.testing.assert_close(lens_loss, kl(end), rtol=1e-4, atol=0.0)
    # Guard against a lens that reads the decomposed position itself: the two KLs must differ.
    at_decomposed = decomposed.float().argmax(dim=1)
    assert (kl(at_decomposed) - kl(end)).abs() > 0.1 * kl(end)
    # The stimulus runs on the original weights: only positions from the decomposed one on change.
    first = int(at_decomposed.min())
    torch.testing.assert_close(masked_logits[:, :first], target_logits[:, :first])


def test_residual_lens_matches_hidden_states() -> None:
    model = _tiny_model(decomposed_layers=(1,))
    ids = torch.randint(1, 1000, (2, 7), generator=torch.Generator().manual_seed(0))
    d_model = model.target_model.get_parameter("model.layers.0.input_layernorm.weight").shape[0]
    std = torch.rand(d_model, generator=torch.Generator().manual_seed(1)) + 0.5
    with torch.no_grad():
        # hidden_states[0] is the embedding output, so hidden_states[1] is after layer 0.
        hidden = model.target_model(ids, output_hidden_states=True).hidden_states[1]
        with model.output_lens(ResidualOutputLens("model.layers.0", std)):
            lens_out = model(ids)
    torch.testing.assert_close(lens_out, hidden / std)


def test_joint_tokenization_counts_the_merged_period_as_prompt() -> None:
    """With `tokenize_jointly`, the batch holds exactly the joint tokenization of prompt + stimulus;
    the token holding the prompt's last character (Qwen merges '.' + '\\n' into '.\\n') is the
    prompt's last token; `TokenPositions(start=1)` selects the prompt's tokens without the first."""
    tok = AutoTokenizer.from_pretrained(TINY_QWEN)
    stimulus = "\nThe amount of truth in this statement is "
    lens = StimulusLens(type="stimulus", stimulus=stimulus, tokenize_jointly=True)
    batch = load_prompts_dataset(
        PROMPTS, tok, MAX_SEQ_LEN, TokenPositions(type="tokens", start=1), lens
    )[:]
    for row, prompt in enumerate(PROMPTS):
        joint = tok.encode(prompt + stimulus, add_special_tokens=False)
        own = tok.encode(prompt, add_special_tokens=False)
        assert batch["input_ids"][row, : len(joint)].tolist() == joint
        assert int(batch[STIMULUS_END_KEY][row]) == len(joint) - 1
        n_prompt = len(own)  # '.' + '\n' merge, so the prompt keeps its token count
        # '.\n', or "'.\n" after a quote
        assert tok.decode([joint[n_prompt - 1]]).endswith(".\n"), tok.decode([joint[n_prompt - 1]])
        assert joint[: n_prompt - 1] == own[:-1]
        selected = batch[POSITION_MASK_KEY][row].nonzero().flatten().tolist()
        assert selected == list(range(1, n_prompt)), selected


def test_joint_tokenization_allows_a_resplit_prompt_end() -> None:
    """A prompt whose end the stimulus re-splits beyond its last token (an OOD statement of tiu v3,
    `facts_conj`) is accepted: its positions are exactly the joint tokens holding its characters,
    and the ordinary prompts in the same call are unchanged."""
    tok = AutoTokenizer.from_pretrained(TINY_QWEN)
    stimulus = "\nThe amount of truth in this statement is "
    resplit = "The planet Uranus is often referred to as the 'ice giant.'."
    prompts = [*PROMPTS, resplit]
    joint = tok(resplit + stimulus, return_offsets_mapping=True, add_special_tokens=False)
    own = tok.encode(resplit, add_special_tokens=False)
    n_prompt = 1 + max(i for i, (s, _) in enumerate(joint["offset_mapping"]) if s < len(resplit))
    assert joint["input_ids"][: n_prompt - 1] != own[: n_prompt - 1], "no re-split: tests nothing"
    lens = StimulusLens(type="stimulus", stimulus=stimulus, tokenize_jointly=True)
    loss_positions = TokenPositions(type="tokens", start=1)
    batch = load_prompts_dataset(prompts, tok, 48, loss_positions, lens)[:]
    alone = load_prompts_dataset(PROMPTS, tok, 48, loss_positions, lens)[:]
    for key in ("input_ids", POSITION_MASK_KEY, STIMULUS_END_KEY):
        assert torch.equal(batch[key][: len(PROMPTS)], alone[key]), key
    row = len(PROMPTS)
    assert batch["input_ids"][row, : len(joint["input_ids"])].tolist() == joint["input_ids"]
    selected = batch[POSITION_MASK_KEY][row].nonzero().flatten().tolist()
    assert selected == list(range(1, n_prompt)), selected
    assert int(batch[STIMULUS_END_KEY][row]) == len(joint["input_ids"]) - 1
