import pytest

from solver import Solver
from trie import construct_scrabble_lexicon


@pytest.fixture(scope="session", autouse=True)
def construct_lexicon():
    construct_scrabble_lexicon()


def test_basic():
    b = Solver()
    tiles = [ch for ch in 'MKJZUSI']
    b.find_best_move(tiles)
    expected_word = 'MUZJIKS'

    assert b.best_word == expected_word, f"Found incorrect word {b.best_word}"
    assert b.best_score == 82
    assert not b.used_transpose, f"Should never use transpose for the first move."
    assert b.start_coords == (7, 1)

    b.apply_best_move()
    for i in range(len(expected_word)):
        assert b.board[7][1 + i].value == expected_word[i]

    # A few checks on cross-check values
    assert not b.board[7][0].cross_checks_horizontal, \
        f"Expected board[7][0] to have no horizontal cc candidates."
    assert len(b.board[7][0].cross_checks_vertical) == 26, \
        f"Expected board[7][0] to have all vertical cc candidates."
    assert b.board[6][1].cross_checks_vertical == {'A', 'E', 'H', 'M', 'O', 'U'}, \
        "These are the only letters which could form a 2-letter word with M (only adjacent filled-space vertically)."

    # Previously, 'MAX' was found (downwards) however it is better to form AX on top of MU with X as DLS
    tiles = [ch for ch in 'AX']
    b.find_best_move(tiles)
    expected_word = 'AX'

    assert b.best_word == expected_word, f"Found incorrect word {b.best_word}"
    assert b.best_score == 38
    assert not b.used_transpose, f"Should *not* use transpose to construct second move."
    assert b.start_coords == (6, 1)  # Inverted because transpose

    used_tiles = b.apply_best_move()
    assert sorted(used_tiles) == ['A', 'X']
    for i in range(len(expected_word)):
        assert b.board[6][1+i].value == expected_word[i]
