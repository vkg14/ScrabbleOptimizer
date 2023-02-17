import pytest

from board import Board
from trie import construct_scrabble_lexicon


@pytest.fixture(scope="session", autouse=True)
def construct_lexicon(request):
    construct_scrabble_lexicon()


def test_initial_move():
    b = Board()
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
