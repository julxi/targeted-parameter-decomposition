"""Test completeness by maximizing circuit divergence under a complement constraint.

Uses greedy coordinate descent with random restarts to optimize binary masks directly,
avoiding the binarization gap of PGD-based optimization.

The "circuit" is the set of active components (CI above threshold). The "complement" is
everything else: inactive components plus weight deltas.

Maximizes Div(original || circuit) subject to Div(original || circuit + complement) < threshold.
A high circuit divergence under this constraint means the complement is needed to recover the
original model's behavior — the circuit alone is incomplete.

Usage:
    # Toy model (TMS/ResidMLP) — input is a feature dimension index:
    python -m spd.scripts.decomposition_stress_test.completeness_test_greedy_constrained \
        <model_path> 3 --complement-threshold 0.01

    # Language model — input is a prompt string:
    python -m spd.scripts.decomposition_stress_test.completeness_test_greedy_constrained \
        <model_path> "Once upon a time" --complement-threshold 0.1
"""

import argparse
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import Config
from spd.experiments.completeness.models import RedundantCopyTransformer
from spd.experiments.resid_mlp.models import ResidMLP
from spd.experiments.tms.models import TMSModel
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.utils.general_utils import calc_kl_divergence_lm, calc_sum_recon_loss_lm


def detect_model_type(config: Config) -> Literal["toy", "lm", "completeness"]:
    match config.task_config.task_name:
        case "completeness":
            return "completeness"
        case "lm":
            return "lm"
        case "tms" | "resid_mlp" | "ih" | "rotgrid":
            return "toy"


def build_input_tensor(
    model: ComponentModel,
    input_str: str,
    model_type: Literal["toy", "lm", "completeness"],
    tokenizer_name: str | None,
    device: torch.device,
) -> tuple[Tensor, PreTrainedTokenizerBase | None]:
    match model_type:
        case "toy":
            dim_idx = int(input_str)
            assert isinstance(model.target_model, ResidMLP | TMSModel)
            n_features = model.target_model.config.n_features
            assert 0 <= dim_idx < n_features, f"dim_idx {dim_idx} out of range [0, {n_features})"
            input_tensor = torch.zeros(1, n_features, device=device)
            input_tensor[0, dim_idx] = 0.75
            return input_tensor, None
        case "completeness":
            token_value = int(input_str)
            assert isinstance(model.target_model, RedundantCopyTransformer)
            cfg = model.target_model.config
            assert 1 <= token_value < cfg.vocab_size, (
                f"token value {token_value} out of range [1, {cfg.vocab_size})"
            )
            input_tensor = torch.tensor([[token_value, cfg.eq_token]], device=device)
            return input_tensor, None
        case "lm":
            assert tokenizer_name is not None, "tokenizer_name required for LM models"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            assert isinstance(tokenizer, PreTrainedTokenizerBase)
            tokens = tokenizer.encode(input_str, return_tensors="pt")
            assert isinstance(tokens, Tensor)
            return tokens.to(device), tokenizer


def compute_ci(model: ComponentModel, input_tensor: Tensor) -> dict[str, Float[Tensor, "... C"]]:
    output_with_cache = model(input_tensor, cache_type="input")
    ci = model.calc_causal_importances(
        pre_weight_acts=output_with_cache.cache,
        sampling="continuous",
    ).upper_leaky
    return ci


def get_circuit_masks(
    ci: dict[str, Float[Tensor, "... C"]], ci_thr: float
) -> dict[str, Float[Tensor, "... C"]]:
    return {module: (ci_vals > ci_thr).float() for module, ci_vals in ci.items()}


def eval_divergences(
    model: ComponentModel,
    input_tensor: Tensor,
    sources: dict[str, Tensor],
    circuit: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    loss_type: Literal["mse", "kl"],
    original_output: Tensor,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Tensor, Tensor]:
    """Returns (div_circuit, div_with_complement, circuit_out, with_complement_out)."""
    circuit_mask = {m: circuit[m] * sources[m] for m in circuit}
    with_complement_mask = {m: circuit[m] * sources[m] + (1 - circuit[m]) for m in circuit}

    wdam: dict[str, WeightDeltaAndMask] = {
        m: (weight_deltas[m], torch.ones(circuit[m].shape[:-1], device=circuit[m].device))
        for m in circuit
    }

    circuit_out = model(input_tensor, mask_infos=make_mask_infos(circuit_mask))
    with_complement_out = model(
        input_tensor,
        mask_infos=make_mask_infos(with_complement_mask, weight_deltas_and_masks=wdam),
    )

    n_examples = circuit_out.shape[:-1].numel() if loss_type == "kl" else circuit_out.numel()
    div_circuit = (
        calc_sum_recon_loss_lm(pred=circuit_out, target=original_output, loss_type=loss_type)
        / n_examples
    )
    div_with_complement = (
        calc_sum_recon_loss_lm(
            pred=with_complement_out, target=original_output, loss_type=loss_type
        )
        / n_examples
    )
    return div_circuit, div_with_complement, circuit_out, with_complement_out


def _per_element_divergences(
    circuit_out: Tensor,
    with_complement_out: Tensor,
    original_output: Tensor,
    loss_type: Literal["mse", "kl"],
    n_batch: int,
) -> tuple[Float[Tensor, " n_batch"], Float[Tensor, " n_batch"]]:
    """Compute per-batch-element divergences for both circuit and circuit+complement."""
    original_expanded = original_output.expand_as(circuit_out)
    match loss_type:
        case "mse":
            div_circuit = ((circuit_out - original_expanded) ** 2).flatten(1).mean(1)
            div_with_complement = (
                ((with_complement_out - original_expanded) ** 2).flatten(1).mean(1)
            )
        case "kl":
            p_orig = torch.softmax(original_expanded, dim=-1)
            log_q_circuit = torch.log_softmax(circuit_out, dim=-1)
            kl_circuit = F.kl_div(log_q_circuit, p_orig, reduction="none").sum(dim=-1)
            div_circuit = kl_circuit.reshape(n_batch, -1).mean(1)
            log_q_with_complement = torch.log_softmax(with_complement_out, dim=-1)
            kl_with_complement = F.kl_div(log_q_with_complement, p_orig, reduction="none").sum(
                dim=-1
            )
            div_with_complement = kl_with_complement.reshape(n_batch, -1).mean(1)
    return div_circuit, div_with_complement


def run_greedy(
    model: ComponentModel,
    input_tensor: Tensor,
    original_output: Tensor,
    circuit: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    loss_type: Literal["mse", "kl"],
    n_restarts: int,
    complement_threshold: float,
) -> tuple[dict[str, Tensor], Float[Tensor, ""], Float[Tensor, ""], Tensor, Tensor]:
    """Greedy coordinate descent maximizing Div(orig, circuit) s.t. Div(orig, circuit+complement) < threshold.

    Returns (best_sources, best_div_circuit, best_div_complement, circuit_out, with_complement_out).
    """
    modules = list(circuit.keys())
    coords: list[tuple[str, int]] = []
    for m in modules:
        flat_circuit = circuit[m].reshape(-1)
        for i in range(flat_circuit.numel()):
            if flat_circuit[i] > 0:
                coords.append((m, i))

    n_coords = len(coords)
    print(f"  {n_coords} circuit coordinates")

    best_div_circuit_val = float("-inf")
    best_div_circuit_t: Float[Tensor, ""] | None = None
    best_div_complement_t: Float[Tensor, ""] | None = None
    best_sources: dict[str, Tensor] | None = None
    best_circuit_out: Tensor | None = None
    best_with_complement_out: Tensor | None = None

    MAX_INIT_ATTEMPTS = 20

    for restart in range(n_restarts):
        # Find a feasible starting point
        sources: dict[str, Tensor] = {m: torch.ones_like(circuit[m]) for m in modules}
        div_circuit = torch.tensor(0.0)
        div_complement = torch.tensor(0.0)
        feasible = False
        for attempt in range(MAX_INIT_ATTEMPTS):
            if attempt < MAX_INIT_ATTEMPTS - 1:
                sources = {
                    m: torch.randint(0, 2, circuit[m].shape, device=circuit[m].device).float()
                    for m in modules
                }
            else:
                # Fallback: all ones (full circuit active, most likely to be feasible)
                sources = {m: torch.ones_like(circuit[m]) for m in modules}

            with torch.no_grad():
                div_circuit, div_complement, _, _ = eval_divergences(
                    model, input_tensor, sources, circuit, weight_deltas, loss_type, original_output
                )

            if div_complement.item() < complement_threshold:
                feasible = True
                break

        if not feasible:
            print(
                f"  Restart {restart + 1}/{n_restarts}: no feasible start found"
                f" (best complement div = {div_complement.item():.6f}), skipping"
            )
            continue

        current_div_circuit = div_circuit

        n_sweeps = 0
        while n_coords > 0:
            n_sweeps += 1

            batched_sources = {
                m: sources[m].expand(n_coords, *sources[m].shape[1:]).clone() for m in modules
            }
            for i, (m, flat_idx) in enumerate(coords):
                batched_sources[m][i].reshape(-1)[flat_idx] = (
                    1.0 - batched_sources[m][i].reshape(-1)[flat_idx]
                )

            batched_circuit = {
                m: circuit[m].expand(n_coords, *circuit[m].shape[1:]) for m in modules
            }
            circuit_mask = {m: batched_circuit[m] * batched_sources[m] for m in modules}
            with_complement_mask = {
                m: batched_circuit[m] * batched_sources[m] + (1 - batched_circuit[m])
                for m in modules
            }
            wdam: dict[str, WeightDeltaAndMask] = {
                m: (
                    weight_deltas[m],
                    torch.ones(batched_circuit[m].shape[:-1], device=circuit[m].device),
                )
                for m in modules
            }
            batched_input = input_tensor.expand(n_coords, *input_tensor.shape[1:])

            with torch.no_grad():
                circuit_out = model(batched_input, mask_infos=make_mask_infos(circuit_mask))
                with_complement_out = model(
                    batched_input,
                    mask_infos=make_mask_infos(with_complement_mask, weight_deltas_and_masks=wdam),
                )

            per_div_circuit, per_div_complement = _per_element_divergences(
                circuit_out, with_complement_out, original_output, loss_type, n_coords
            )

            # Accept flips that improve div_circuit AND stay feasible
            applied_flips: list[tuple[str, int]] = []
            for i, (m, flat_idx) in enumerate(coords):
                if (
                    per_div_circuit[i] > current_div_circuit
                    and per_div_complement[i] < complement_threshold
                ):
                    sources[m].reshape(-1)[flat_idx] = 1.0 - sources[m].reshape(-1)[flat_idx]
                    applied_flips.append((m, flat_idx))

            if not applied_flips:
                break

            # Re-evaluate after all flips applied together
            with torch.no_grad():
                new_div_circuit, new_div_complement, _, _ = eval_divergences(
                    model, input_tensor, sources, circuit, weight_deltas, loss_type, original_output
                )

            if new_div_circuit > current_div_circuit and new_div_complement < complement_threshold:
                current_div_circuit = new_div_circuit
            else:
                for m, flat_idx in applied_flips:
                    sources[m].reshape(-1)[flat_idx] = 1.0 - sources[m].reshape(-1)[flat_idx]
                break

        print(
            f"  Restart {restart + 1}/{n_restarts}:"
            f" div_circuit = {current_div_circuit.item():.6f} ({n_sweeps} sweeps)"
        )

        if current_div_circuit.item() > best_div_circuit_val:
            best_div_circuit_val = current_div_circuit.item()
            best_sources = {m: sources[m].clone() for m in modules}
            with torch.no_grad():
                (
                    best_div_circuit_t,
                    best_div_complement_t,
                    best_circuit_out,
                    best_with_complement_out,
                ) = eval_divergences(
                    model, input_tensor, sources, circuit, weight_deltas, loss_type, original_output
                )

    assert best_div_circuit_t is not None, "No feasible solution found across all restarts"
    assert best_div_complement_t is not None
    assert best_sources is not None
    assert best_circuit_out is not None
    assert best_with_complement_out is not None
    return (
        best_sources,
        best_div_circuit_t,
        best_div_complement_t,
        best_circuit_out,
        best_with_complement_out,
    )


def print_results(
    circuit: dict[str, Float[Tensor, "... C"]],
    binary_sources: dict[str, Tensor],
    div_circuit: Float[Tensor, ""],
    div_with_complement: Float[Tensor, ""],
    model_type: Literal["toy", "lm", "completeness"],
    ci_thr: float,
    complement_threshold: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Completeness Test Results (CI threshold = {ci_thr})")
    print(f"{'=' * 60}")
    print(f"Complement threshold: {complement_threshold}")
    print(f"Best Div(original, circuit):              {div_circuit.item():.6f}")
    print(f"Best Div(original, circuit + complement): {div_with_complement.item():.6f}")

    for module in circuit:
        circuit_mask = circuit[module]
        binary = binary_sources[module]

        print(f"\nModule: {module}")

        match model_type:
            case "toy":
                circuit_indices = circuit_mask[0].nonzero(as_tuple=True)[0].tolist()
                unmasked = ((circuit_mask[0] * binary[0]) > 0.5).nonzero(as_tuple=True)[0].tolist()
                print(f"  Circuit components: {circuit_indices}, unmasked: {unmasked}")
            case "lm" | "completeness":
                for pos in range(circuit_mask.shape[1]):
                    circuit_indices = circuit_mask[0, pos].nonzero(as_tuple=True)[0].tolist()
                    unmasked = (
                        ((circuit_mask[0, pos] * binary[0, pos]) > 0.5)
                        .nonzero(as_tuple=True)[0]
                        .tolist()
                    )
                    print(
                        f"  Position {pos}: circuit components: {circuit_indices},"
                        f" unmasked: {unmasked}"
                    )


def print_divergences(
    original: Tensor,
    circuit_out: Tensor,
    with_complement_out: Tensor,
    loss_type: Literal["mse", "kl"],
) -> None:
    print(f"\n{'=' * 60}")
    print("Divergence from Original Model Output")
    print(f"{'=' * 60}")

    match loss_type:
        case "mse":
            mse_circuit = ((original - circuit_out) ** 2).mean().item()
            mse_with_complement = ((original - with_complement_out) ** 2).mean().item()
            print(f"  MSE(original, circuit)              = {mse_circuit:.6f}")
            print(f"  MSE(original, circuit + complement) = {mse_with_complement:.6f}")
        case "kl":
            kl_circuit = calc_kl_divergence_lm(pred=circuit_out, target=original).item()
            kl_with_complement = calc_kl_divergence_lm(
                pred=with_complement_out, target=original
            ).item()
            print(f"  KL(original, circuit)              = {kl_circuit:.6f}")
            print(f"  KL(original, circuit + complement) = {kl_with_complement:.6f}")


def print_top_outputs(
    original: Tensor,
    circuit_out: Tensor,
    with_complement_out: Tensor,
    model_type: Literal["toy", "lm", "completeness"],
    tokenizer: PreTrainedTokenizerBase | None,
    k: int = 10,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Top-{k} Output Dimensions/Tokens")
    print(f"{'=' * 60}")

    match model_type:
        case "toy":
            _, indices = torch.topk(original[0].abs(), k=min(k, original.shape[-1]))
            print(f"  {'Dim':<6} {'Original':>10} {'Circuit':>10} {'+Complement':>12}")
            print(f"  {'-' * 42}")
            for idx in indices:
                i = int(idx)
                print(
                    f"  {i:<6} {original[0, i].item():>10.4f}"
                    f" {circuit_out[0, i].item():>10.4f}"
                    f" {with_complement_out[0, i].item():>12.4f}"
                )
        case "completeness":
            # Output is [batch, vocab_size] (model returns logits at the prediction position)
            probs_orig = torch.softmax(original[0], dim=-1)
            probs_circuit = torch.softmax(circuit_out[0], dim=-1)
            probs_with_complement = torch.softmax(with_complement_out[0], dim=-1)

            _, indices = torch.topk(probs_orig, k=min(k, probs_orig.shape[-1]))
            print(f"  {'Token idx':<12} {'Original':>10} {'Circuit':>10} {'+Complement':>12}")
            print(f"  {'-' * 48}")
            for idx in indices:
                i = int(idx)
                print(
                    f"  {i:<12} {probs_orig[i].item():>10.4f}"
                    f" {probs_circuit[i].item():>10.4f}"
                    f" {probs_with_complement[i].item():>12.4f}"
                )
        case "lm":
            assert tokenizer is not None
            seq_len = original.shape[1]
            for pos in range(seq_len):
                probs_orig = torch.softmax(original[0, pos], dim=-1)
                probs_circuit = torch.softmax(circuit_out[0, pos], dim=-1)
                probs_with_complement = torch.softmax(with_complement_out[0, pos], dim=-1)

                print(f"\n  Position {pos}:")
                _, indices = torch.topk(probs_orig, k=min(k, probs_orig.shape[-1]))
                print(f"  {'Token':<20} {'Original':>10} {'Circuit':>10} {'+Complement':>12}")
                print(f"  {'-' * 56}")
                for idx in indices:
                    i = int(idx)
                    token_str = repr(tokenizer.decode([i]))
                    print(
                        f"  {token_str:<20} {probs_orig[i].item():>10.4f}"
                        f" {probs_circuit[i].item():>10.4f}"
                        f" {probs_with_complement[i].item():>12.4f}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPD Completeness Test (Greedy Constrained Optimization)"
    )
    parser.add_argument("model_path", help="Path to decomposed model (wandb or local)")
    parser.add_argument("input", help="Prompt string (LM) or dimension index (toy model)")
    parser.add_argument(
        "--complement-threshold",
        type=float,
        required=True,
        help="Max allowed Div(original, circuit + complement)",
    )
    parser.add_argument(
        "--ci-thr", type=float, default=0.01, help="CI threshold for circuit components"
    )
    parser.add_argument("--n-restarts", type=int, default=10, help="Number of random restarts")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top output dims to show")
    parser.add_argument("--device", default="cpu", help="Device to run on")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load model and config
    print(f"Loading model from {args.model_path}...")
    run_info = SPDRunInfo.from_path(args.model_path)
    config = run_info.config
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    model_type = detect_model_type(config)
    print(f"Model type: {model_type} (task={config.task_config.task_name})")

    # 2. Construct input tensor
    input_tensor, tokenizer = build_input_tensor(
        model, args.input, model_type, config.tokenizer_name, device
    )

    # 3. Compute CI values and identify circuit components
    ci = compute_ci(model, input_tensor)
    circuit = get_circuit_masks(ci, args.ci_thr)
    n_circuit = sum(int(mask.any(dim=-1).sum()) for mask in circuit.values())
    print(f"Computing causal importances... {n_circuit} circuit components")

    # 4. Compute weight deltas
    weight_deltas = model.calc_weight_deltas()

    # 5. Compute original model output (no masks = target model behavior)
    with torch.no_grad():
        original_output = model(input_tensor)

    # 6. Greedy coordinate descent optimization
    print(
        f"\nRunning greedy coordinate descent ({args.n_restarts} restarts,"
        f" complement threshold = {args.complement_threshold})..."
    )
    binary_sources, div_circuit, div_complement, circuit_out, with_complement_out = run_greedy(
        model=model,
        input_tensor=input_tensor,
        original_output=original_output,
        circuit=circuit,
        weight_deltas=weight_deltas,
        loss_type=config.output_loss_type,
        n_restarts=args.n_restarts,
        complement_threshold=args.complement_threshold,
    )

    # 7. Print results
    print_results(
        circuit,
        binary_sources,
        div_circuit,
        div_complement,
        model_type,
        args.ci_thr,
        args.complement_threshold,
    )
    print_divergences(original_output, circuit_out, with_complement_out, config.output_loss_type)
    print_top_outputs(
        original_output, circuit_out, with_complement_out, model_type, tokenizer, args.top_n
    )


if __name__ == "__main__":
    main()
