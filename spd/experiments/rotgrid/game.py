"""Rotating 4x3 grid-world: the rules, and the finite automaton they induce.

Board, with row 0 at the top and ``=`` marking a wall between two cells::

    +-------+-------+-------+
    | A     | empty | B     |
    +=======+-------+=======+
    | empty | start | empty |
    +-------+-------+-------+
    | empty | empty | empty |
    +-------+-------+-------+
    | X     | X     | X     |
    +-------+-------+-------+

A run begins with ``new`` (the robot appears on ``start``) and ends when the robot reaches a goal.
Reaching ``A`` turns the board a quarter turn anticlockwise for the next run; reaching ``B`` turns it
a quarter turn clockwise; reaching an ``X`` leaves it as it is. Moves are named in the viewer's
frame, so the board's rotation changes which moves lead where.

Walls seal ``A`` and ``B`` off from the row below, so the only way into either is sideways out of the
cell between them. The last move of a run is therefore what decides between them.

The board is a rigid object: a quarter turn carries its walls and its edges around with it, and
leaves it standing three wide and four tall. Nothing in the rules consults that footprint, so the
changing aspect ratio never has to be represented.

The robot may never move into a wall or off the board, so the legal tokens at any moment are
determined by the cell it stands on and the board's accumulated rotation. That pair is the world
state, and there are only 48 of them.
"""

import torch
from jaxtyping import Bool, Int
from torch import Tensor

N_ROWS = 4
N_COLS = 3
N_CELLS = N_ROWS * N_COLS
N_ORIENTATIONS = 4
N_STATES = N_CELLS * N_ORIENTATIONS

TOKENS = ("up", "left", "right", "down", "new")
N_TOKENS = len(TOKENS)
NEW = TOKENS.index("new")
MOVE_TOKENS = TOKENS[:NEW]

Cell = tuple[int, int]

START: Cell = (1, 1)
GOALS: dict[Cell, int] = {(0, 0): +1, (0, 2): -1, (3, 0): 0, (3, 1): 0, (3, 2): 0}
WALLS = frozenset(
    {
        frozenset({(0, 0), (1, 0)}),
        frozenset({(0, 2), (1, 2)}),
    }
)

SCREEN_DELTAS: dict[str, Cell] = {"up": (-1, 0), "left": (0, -1), "right": (0, 1), "down": (1, 0)}

ILLEGAL_TRANSITION = -1


def _quarter_turn_clockwise(delta: Cell) -> Cell:
    d_row, d_col = delta
    return (d_col, -d_row)


def board_delta(token: str, orientation: int) -> Cell:
    """Convert a move named in the viewer's frame into a step in the board's own frame.

    The board has been turned `orientation` quarter turns anticlockwise, so undoing that means
    turning the move `orientation` quarter turns clockwise.
    """
    delta = SCREEN_DELTAS[token]
    for _ in range(orientation):
        delta = _quarter_turn_clockwise(delta)
    return delta


def cell_index(cell: Cell) -> int:
    row, col = cell
    return row * N_COLS + col


def cell_from_index(index: int) -> Cell:
    return divmod(index, N_COLS)


def state_index(cell: Cell, orientation: int) -> int:
    return cell_index(cell) * N_ORIENTATIONS + orientation


def state_components(state: int) -> tuple[Cell, int]:
    index, orientation = divmod(state, N_ORIENTATIONS)
    return cell_from_index(index), orientation


INITIAL_STATE = state_index(START, 0)


def legal_tokens(cell: Cell, orientation: int) -> tuple[str, ...]:
    """The tokens that may follow a robot standing on `cell` with the board at `orientation`."""
    if cell in GOALS:
        return ("new",)

    legal = []
    for token in MOVE_TOKENS:
        d_row, d_col = board_delta(token, orientation)
        destination = (cell[0] + d_row, cell[1] + d_col)
        on_board = 0 <= destination[0] < N_ROWS and 0 <= destination[1] < N_COLS
        if on_board and frozenset({cell, destination}) not in WALLS:
            legal.append(token)
    return tuple(legal)


def apply_token(cell: Cell, orientation: int, token: str) -> tuple[Cell, int]:
    """Advance the world state by one token."""
    assert token in legal_tokens(cell, orientation), (
        f"illegal token {token!r} at cell {cell} with orientation {orientation}"
    )
    if token == "new":
        return START, orientation

    d_row, d_col = board_delta(token, orientation)
    destination = (cell[0] + d_row, cell[1] + d_col)
    delta = GOALS.get(destination, 0)
    return destination, (orientation + delta) % N_ORIENTATIONS


def _build_automaton() -> tuple[
    Bool[Tensor, "n_states n_tokens"], Int[Tensor, "n_states n_tokens"]
]:
    legal_mask = torch.zeros(N_STATES, N_TOKENS, dtype=torch.bool)
    next_state = torch.full((N_STATES, N_TOKENS), ILLEGAL_TRANSITION, dtype=torch.long)

    for state in range(N_STATES):
        cell, orientation = state_components(state)
        for token in legal_tokens(cell, orientation):
            token_index = TOKENS.index(token)
            legal_mask[state, token_index] = True
            next_state[state, token_index] = state_index(*apply_token(cell, orientation, token))

    return legal_mask, next_state


LEGAL_MASK, NEXT_STATE = _build_automaton()


def reachable_states() -> set[int]:
    """Every world state reachable from `INITIAL_STATE`."""
    seen: set[int] = set()
    frontier = [INITIAL_STATE]
    while frontier:
        state = frontier.pop()
        if state in seen:
            continue
        seen.add(state)
        cell, orientation = state_components(state)
        for token in legal_tokens(cell, orientation):
            frontier.append(state_index(*apply_token(cell, orientation, token)))
    return seen


assert LEGAL_MASK.any(dim=1).all(), "every world state must offer at least one legal token"
assert len(reachable_states()) == N_STATES, "the automaton should have no unreachable states"
