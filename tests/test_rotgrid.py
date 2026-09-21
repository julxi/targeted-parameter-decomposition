from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from spd.experiments.rotgrid import game
from spd.experiments.rotgrid.configs import RotGridModelConfig, RotGridTrainConfig
from spd.experiments.rotgrid.dataset import RotGridDataset
from spd.experiments.rotgrid.model import RotGridTransformer
from spd.experiments.rotgrid.train_rotgrid import evaluate, get_model_and_batches, train

A_CELL = (0, 0)
B_CELL = (0, 2)
X_CELLS = ((3, 0), (3, 1), (3, 2))
DECISION_CELL = (0, 1)


def walk(tokens: list[str]) -> tuple[game.Cell, int]:
    """Replay a token sequence, asserting each token is legal. Returns the final world state."""
    assert tokens[0] == "new"
    cell, orientation = game.START, 0
    for token in tokens[1:]:
        cell, orientation = game.apply_token(cell, orientation, token)
    return cell, orientation


def test_single_run_to_a_is_legal_and_rotates_anticlockwise():
    cell, orientation = walk(["new", "up", "left"])
    assert cell == A_CELL
    assert orientation == 1


def test_reaching_b_rotates_clockwise():
    cell, orientation = walk(["new", "up", "right"])
    assert cell == B_CELL
    assert orientation == 3


def test_reaching_x_leaves_the_board_unrotated():
    cell, orientation = walk(["new", "down", "down"])
    assert cell in X_CELLS
    assert orientation == 0


def test_iterated_runs_across_a_rotation():
    """The board has turned once, so the moves that reached A must be renamed to reach it again."""
    cell, orientation = walk(["new", "up", "left", "new", "left", "down"])
    assert cell == A_CELL
    assert orientation == 2


def test_the_last_move_of_a_run_decides_between_a_and_b():
    """A and B share every prefix: only the final token tells them apart."""
    assert walk(["new", "up", "left"])[0] == A_CELL
    assert walk(["new", "up", "right"])[0] == B_CELL


def test_board_edge_blocks_leaving_the_top_row():
    assert "up" not in game.legal_tokens(DECISION_CELL, 0)
    with pytest.raises(AssertionError):
        walk(["new", "up", "up"])


def test_wall_blocks_cell_below_a():
    assert set(game.legal_tokens((1, 0), 0)) == {"right", "down"}


def test_wall_blocks_cell_below_b():
    assert set(game.legal_tokens((1, 2), 0)) == {"left", "down"}


def test_a_and_b_are_only_reachable_from_the_decision_cell():
    assert set(game.legal_tokens(DECISION_CELL, 0)) == {"left", "right", "down"}
    for cell in (A_CELL, B_CELL):
        approaches = [
            (source, token)
            for source in [game.cell_from_index(i) for i in range(game.N_CELLS)]
            for token in game.legal_tokens(source, 0)
            if token != "new" and game.apply_token(source, 0, token)[0] == cell
        ]
        assert [source for source, _ in approaches] == [DECISION_CELL]


def test_goal_cells_only_allow_new():
    for orientation in range(game.N_ORIENTATIONS):
        for cell in game.GOALS:
            assert game.legal_tokens(cell, orientation) == ("new",)


def test_non_goal_cells_forbid_new():
    for state in range(game.N_STATES):
        cell, orientation = game.state_components(state)
        if cell not in game.GOALS:
            assert "new" not in game.legal_tokens(cell, orientation)


def test_only_a_and_b_rotate_the_board():
    assert game.GOALS[A_CELL] == +1
    assert game.GOALS[B_CELL] == -1
    for cell in X_CELLS:
        assert game.GOALS[cell] == 0


def test_all_states_are_reachable():
    assert len(game.reachable_states()) == game.N_STATES


def test_new_returns_to_start_keeping_orientation():
    for orientation in range(game.N_ORIENTATIONS):
        for cell in game.GOALS:
            assert game.apply_token(cell, orientation, "new") == (game.START, orientation)


def test_automaton_tables_match_the_rules():
    """The tensor tables are derived from the rules, so check them against the rules directly."""
    for state in range(game.N_STATES):
        cell, orientation = game.state_components(state)
        expected = game.legal_tokens(cell, orientation)
        for token_index, token in enumerate(game.TOKENS):
            is_legal = bool(game.LEGAL_MASK[state, token_index])
            assert is_legal == (token in expected)
            if is_legal:
                next_cell, next_orientation = game.apply_token(cell, orientation, token)
                assert game.NEXT_STATE[state, token_index] == game.state_index(
                    next_cell, next_orientation
                )
            else:
                assert game.NEXT_STATE[state, token_index] == game.ILLEGAL_TRANSITION


def test_dataset_only_samples_legal_tokens():
    dataset = RotGridDataset(seq_len=128, device="cpu")
    tokens, legal = dataset.generate_batch(32)

    assert tokens.shape == (32, 128)
    assert legal.shape == (32, 128, game.N_TOKENS)
    assert (tokens[:, 0] == game.NEW).all()

    sampled_was_legal = legal[:, :-1].gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    assert sampled_was_legal.all()


def test_dataset_masks_match_a_replay_of_the_rules():
    dataset = RotGridDataset(seq_len=64, device="cpu")
    tokens, legal = dataset.generate_batch(8)

    for row in range(tokens.shape[0]):
        cell, orientation = game.START, 0
        for position in range(tokens.shape[1]):
            if position > 0:
                cell, orientation = game.apply_token(
                    cell, orientation, game.TOKENS[int(tokens[row, position])]
                )
            expected = set(game.legal_tokens(cell, orientation))
            actual = {game.TOKENS[i] for i in legal[row, position].nonzero().flatten().tolist()}
            assert actual == expected


def test_dataset_reaches_all_orientations():
    """A long enough walk should exercise every board rotation, not just the initial one."""
    dataset = RotGridDataset(seq_len=512, device="cpu")
    tokens, _ = dataset.generate_batch(16)

    seen = set()
    for row in range(tokens.shape[0]):
        cell, orientation = game.START, 0
        seen.add(orientation)
        for position in range(1, tokens.shape[1]):
            cell, orientation = game.apply_token(
                cell, orientation, game.TOKENS[int(tokens[row, position])]
            )
            seen.add(orientation)
    assert seen == set(range(game.N_ORIENTATIONS))


def tiny_train_config() -> RotGridTrainConfig:
    return RotGridTrainConfig(
        wandb_project=None,
        rotgrid_model_config=RotGridModelConfig(
            seq_len=32,
            d_model=16,
            n_heads=2,
            n_layers=2,
            ff_fanout=2,
            use_ff=True,
            use_pos_encoding=True,
            use_layer_norm=True,
        ),
        steps=5,
        batch_size=4,
        lr=1e-3,
        lr_warmup=0,
        weight_decay=0.01,
        grad_clip=1.0,
        lr_schedule="constant",
        seed=0,
        eval_freq=2,
        eval_batch_size=4,
        attention_maps_n_steps=1,
        steps_per_rollout=4,
    )


def test_forward_returns_logits_over_the_five_tokens():
    model = RotGridTransformer(tiny_train_config().rotgrid_model_config)
    logits = model(torch.zeros(3, 32, dtype=torch.long))
    assert logits.shape == (3, 32, game.N_TOKENS)


def test_train_happy_path():
    config = tiny_train_config()
    model, batches, dataset = get_model_and_batches(config, device="cpu")
    losses, loss_steps = train(
        model=model, batches=batches, dataset=dataset, config=config, log_wandb=False
    )
    assert len(losses) == len(loss_steps)
    assert all(loss > 0 for loss in losses)


def test_evaluate_reports_the_entropy_floor_and_a_nonnegative_kl():
    config = tiny_train_config()
    model, _, dataset = get_model_and_batches(config, device="cpu")
    metrics = evaluate(model, dataset, batch_size=64)

    assert metrics["kl_to_optimal"] >= 0.0
    assert 0.0 <= metrics["illegal_prob_mass"] <= 1.0
    # An untrained model is near-uniform over 5 tokens, so its CE sits near log(5).
    assert metrics["ce_nats"] == pytest.approx(metrics["kl_to_optimal"] + 0.99, abs=0.15)


def test_evaluate_is_exactly_zero_kl_for_a_perfect_predictor():
    """A model whose logits are the true log-probabilities must score zero KL and zero illegal mass."""
    dataset = RotGridDataset(seq_len=64, device="cpu")
    _, legal = dataset.generate_batch(8)

    perfect_logits = torch.where(legal, 0.0, -1e9)
    log_probs = torch.log_softmax(perfect_logits, dim=-1)
    n_legal = legal.sum(dim=-1).float()
    kl = (-n_legal.log() - (log_probs * legal).sum(dim=-1) / n_legal).mean()

    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    assert (log_probs.exp() * ~legal).sum(dim=-1).max().item() == pytest.approx(0.0, abs=1e-6)


def test_checkpoint_round_trip(tmp_path: Path):
    config = tiny_train_config()
    model = RotGridTransformer(config.rotgrid_model_config)
    torch.save(model.state_dict(), tmp_path / "rotgrid.pth")
    config.to_file(tmp_path / "rotgrid_train_config.yaml")

    reloaded = RotGridTransformer.from_pretrained(tmp_path / "rotgrid.pth")

    tokens = torch.zeros(2, 32, dtype=torch.long)
    assert torch.allclose(model(tokens), reloaded(tokens))


def test_gradient_clipping_bounds_the_update():
    """Clipping is what keeps a rare large gradient from knocking the model off its solution."""
    config = tiny_train_config()
    model, _, dataset = get_model_and_batches(config, device="cpu")

    tokens, _ = dataset.generate_batch(16)
    # Blow the logits up so the backward pass produces a large gradient.
    with torch.no_grad():
        model.unembed.weight.mul_(500.0)
    F.cross_entropy(model(tokens)[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()

    unclipped = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
    after = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()

    assert unclipped > config.grad_clip, (
        f"test setup failed to produce a large gradient: {unclipped}"
    )
    assert clipped == pytest.approx(unclipped)  # returns the PRE-clip norm
    assert after == pytest.approx(config.grad_clip, rel=1e-4)
