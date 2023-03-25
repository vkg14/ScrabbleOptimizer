import functools
import pickle
from dataclasses import dataclass, field
from enum import auto
from operator import add
from typing import Set, Optional, Tuple, List, Callable

import numpy as np
from strenum import StrEnum
from trie import TrieNode

LETTER_VALUES = {"A": 1,
                 "B": 3,
                 "C": 3,
                 "D": 2,
                 "E": 1,
                 "F": 4,
                 "G": 2,
                 "H": 4,
                 "I": 1,
                 "J": 1,
                 "K": 5,
                 "L": 1,
                 "M": 3,
                 "N": 1,
                 "O": 1,
                 "P": 3,
                 "Q": 10,
                 "R": 1,
                 "S": 1,
                 "T": 1,
                 "U": 1,
                 "V": 4,
                 "W": 4,
                 "X": 8,
                 "Y": 4,
                 "Z": 10,
                 "#": 0}


class SquareType(StrEnum):
    NORMAL = auto()
    DLS = auto()
    TLS = auto()
    DWS = auto()
    TWS = auto()

    def letter_multiplier(self):
        if self == SquareType.DLS:
            return 2
        elif self == SquareType.TLS:
            return 3
        else:
            return 1

    def word_multiplier(self):
        if self == SquareType.DWS:
            return 2
        elif self == SquareType.TWS:
            return 3
        else:
            return 1


@dataclass
class Square:
    value: Optional[str] = None
    cross_sum_vertical: int = 0
    cross_checks_vertical: Set[str] = field(default_factory=set)
    cross_sum_horizontal: int = 0
    cross_checks_horizontal: Set[str] = field(default_factory=set)
    typ: SquareType = SquareType.NORMAL

    def __post_init__(self):
        self.cross_checks_vertical = {letter for letter in LETTER_VALUES if letter != '#'}
        self.cross_checks_horizontal = {letter for letter in LETTER_VALUES if letter != '#'}

    def vacant(self):
        return not self.value

    def set_cross_checks_vertical(self, s: Set[str]):
        self.cross_checks_vertical = s

    def set_cross_checks_horizontal(self, s: Set[str]):
        self.cross_checks_horizontal = s

    def set_cross_sum_vertical(self, val: int):
        self.cross_sum_vertical= val

    def set_cross_sum_horizontal(self, val: int):
        self.cross_sum_horizontal = val

    def __str__(self):
        if self.vacant():
            return "_"
        return self.value

    def __repr__(self):
        placeholder = "_"
        return f"Square({self.typ}, {self.value or placeholder})"


@dataclass
class ScrabbleBoard:
    board: List[List[Square]] = field(default_factory=list)
    # TODO: keep track of words played
    words_played: Set[str] = field(default_factory=set)
    transposed: bool = False
    n: int = 15

    def __post_init__(self):
        self.board = [[Square() for _ in range(self.n)] for _ in range(self.n)]
        self._add_premium_squares()

    def in_bounds(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def _add_premium_squares(self):
        # Adds all premium squares that influence the word's score.
        premium_mapping = {
            SquareType.TWS: [(0, 0), (7, 0), (14, 0), (0, 7), (14, 7), (0, 14), (7, 14), (14, 14)],
            SquareType.DWS: [
                (1, 1), (2, 2), (3, 3), (4, 4), (1, 13), (2, 12), (3, 11), (4, 10), (13, 1), (12, 2), (11, 3), (10, 4),
                (13, 13), (12, 12), (11, 11), (10, 10)
            ],
            SquareType.TLS: [
                (1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13), (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)
            ],
            SquareType.DLS: [
                (0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14), (6, 2), (6, 6), (6, 8), (6, 12), (7, 3),
                (7, 11), (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14), (12, 6), (12, 8), (14, 3),
                (14, 11)
            ]
        }
        for typ, coords in premium_mapping.items():
            for r, c in coords:
                self.board[r][c].typ = typ

    def transpose(self):
        # Could also transpose with [list(x) for x in zip(*a)]
        self.board = np.array(self.board).T.tolist()
        self.transposed = not self.transposed

    def __getitem__(self, idx: int) -> List[Square]:
        return self.board[idx]


class Solver:
    def __init__(self, lexicon_file='lexicon/scrabble_word_list.pickle'):
        self.board = ScrabbleBoard()

        with open(lexicon_file, 'rb') as f:
            self.dict_trie: TrieNode = pickle.load(f)

        # Best word tracking
        self.best_word = ""
        self.start_coords = (-1, -1)
        self.best_score = 0
        self.score_per_letter: float = 0.0
        self.used_transpose = False

    def _compute_checks(self):
        for r in range(self.board.n):
            for c in range(self.board.n):
                if not self.board[r][c].vacant():
                    continue
                adjacents = [(r, c-1), (r, c+1)]
                check = False
                for r1, c1 in adjacents:
                    if not self.board.in_bounds(r1, c1):
                        continue
                    if not self.board[r1][c1].vacant():
                        # at least one filled adjacent square along row
                        check = True
                        break
                if check:
                    self._update_h_cross_check((r, c))

    def fill_scenario(self, scenario: List[List[str]]):
        assert len(scenario) == self.board.n and len(scenario[0]) == self.board.n, \
            f"Wrong size board ({len(scenario)}, {len(scenario[0])})."
        for r in range(self.board.n):
            for c in range(self.board.n):
                if scenario[r][c] != '_':
                    self.board[r][c].value = scenario[r][c]

        self._compute_checks()
        self.board.transpose()
        self._compute_checks()
        self.board.transpose()

    def _is_potential_anchor(self, r, c):
        """
        Every move (except the very first) requires an anchor, a cell which is currently vacant but adjacent to
        at least one occupied cell.  We build potential moves around anchor cells.
        """
        adjacent_squares = [self.board[r2][c2] for r2, c2 in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if
                            self.board.in_bounds(r2, c2)]
        # At least one of the adjacent squares must have a letter to tether to
        return self.board[r][c].vacant() and any(not square.vacant() for square in adjacent_squares)

    def score_word(self, word: str, coords: Tuple[int, int]):
        r, c = coords
        score = 0
        cross_words_score = 0
        word_multiplier = 1
        new_letters = 0
        for i, ch in enumerate(word):
            square = self.board[r][c + i]
            if not square.vacant():
                assert square.value == ch, f"Problem with word, {word}, formed at {coords} in {i}th char ({ch})."
                score += LETTER_VALUES[ch]
            else:
                lm = square.typ.letter_multiplier()
                wm = square.typ.word_multiplier()
                score += (LETTER_VALUES[ch] * lm)
                word_multiplier *= wm
                new_letters += 1
                # Solve for cross words
                cross_sum = square.cross_sum_vertical if not self.board.transposed else square.cross_sum_horizontal
                if cross_sum > 0:
                    cross_words_score += (cross_sum + LETTER_VALUES[ch] * lm) * wm
        score *= word_multiplier
        score += cross_words_score
        if new_letters == 7:
            score += 50
        # Pick word that gives most points per letter placed
        avg_score = score / new_letters
        if avg_score > self.score_per_letter:
            self.best_word = word
            self.best_score = score
            self.start_coords = coords
            self.used_transpose = self.board.transposed
            self.score_per_letter = avg_score
        return score

    def extend_right(
            self,
            partial_word: str,
            trie_node: TrieNode, tiles: List[str],
            coords: Tuple[int, int],
            anchor_placed: bool = True
    ):
        r, c = coords
        if not self.board.in_bounds(r, c):
            # Since we are out of bounds, no further exploration can happen
            # Previously, we checked if we had no tiles here but we MUST continue until end of board or vacancy
            if trie_node.is_valid_word and anchor_placed:
                self.score_word(partial_word, (r, c - len(partial_word)))
            return
        square = self.board[r][c]
        if square.vacant():
            if trie_node.is_valid_word and anchor_placed:
                self.score_word(partial_word, (r, c - len(partial_word)))
                pass
            for letter, child_node in trie_node.children.items():
                # We always form words horizontally but change the orientation of the board repr beforehand.
                cross_checks = square.cross_checks_vertical \
                    if not self.board.transposed else square.cross_checks_horizontal
                if letter not in tiles or letter not in cross_checks:
                    continue
                tiles.remove(letter)
                self.extend_right(partial_word + letter, child_node, tiles, (r, c + 1))
                tiles.append(letter)
        else:
            # square already occupied by letter
            letter = square.value
            if letter in trie_node.children:
                child_node = trie_node.children[letter]
                self.extend_right(partial_word + letter, child_node, tiles, (r, c + 1))

    def left_part(self, partial_word: str, trie_node: TrieNode, tiles: List[str], limit: int, anchor: Tuple[int, int]):
        # The left part is guaranteed to have no cross-checks so we follow trie based on tiles
        self.extend_right(partial_word, trie_node, tiles, anchor, anchor_placed=False)
        if limit > 0:
            for letter, child_node in trie_node.children.items():
                if letter not in tiles:
                    continue
                tiles.remove(letter)
                self.left_part(partial_word + letter, child_node, tiles, limit - 1, anchor)
                tiles.append(letter)

    def _best_move_helper(self, tiles: List[str]):
        # When in the middle of game, solve using anchors
        for r in range(self.board.n):
            # We solve the best for each row
            # We must keep track of the number of vacant / unconnected squares left of anchor
            left_of_anchor = 0
            prefix = ""
            for c in range(self.board.n):
                if not self.board[r][c].vacant():
                    # If prefix is non-empty, left_of_anc = 0
                    prefix += self.board[r][c].value
                    left_of_anchor = 0
                    continue
                if not self._is_potential_anchor(r, c):
                    # vacant non-anchor -> increment left and clear prefix
                    prefix = ""
                    left_of_anchor += 1
                    continue
                # Found an anchor, run algorithm with built prefix
                start_node = self.dict_trie.traverse_prefix(prefix)
                if start_node:
                    # We should NEVER error here since an existing prefix should be valid
                    self.left_part(prefix, start_node, tiles, left_of_anchor, (r, c))
                # Since this anchor was both vacant and connected, prefix and left must be reset.
                prefix = ""
                left_of_anchor = 0

    def find_best_move(self, tiles: List[str]):
        self._clear_turn_state()
        if self.board[7][7].vacant():
            # start of game, apply best move horizontally since symmetric
            self.left_part("", self.dict_trie, tiles, 6, (7, 7))
            return
        self._best_move_helper(tiles)
        # Solve for vertically formed words using transpose
        self.board.transpose()
        self._best_move_helper(tiles)
        # Undo transpose
        self.board.transpose()

    def _clear_turn_state(self):
        self.best_score = 0
        self.score_per_letter = 0
        self.best_word = ""
        self.start_coords = (-1, -1)
        self.used_transpose = False
        if self.board.transposed:
            self.board.transpose()

    def _find_first_vacant_in_column(self, r: int, c: int, direction: Callable):
        r = direction(r)
        while self.board.in_bounds(r, c) and not self.board[r][c].vacant():
            r = direction(r)
        if self.board.in_bounds(r, c):
            return r, c
        return None

    def _update_v_cross_check(self, coords: Tuple[int, int]):
        """
        This method can be broken down into 2 phases.
        1. Track up and find the prefix above the cross-check coords.
        2. Find whether any current letter in existing cross-check leads to a valid word
        with the computed prefix (above) and computed suffix (walking down column).
        """
        r, c = coords
        upper = r
        while self.board.in_bounds(upper - 1, c) and not self.board[upper - 1][c].vacant():
            upper -= 1
        prefix_node = self.dict_trie
        prefix_sum = 0
        while upper < r:
            # TODO: Exception check since prefix should always exist
            prefix_node = prefix_node.children[self.board[upper][c].value]
            prefix_sum += LETTER_VALUES[self.board[upper][c].value]
            upper += 1
        # We must refresh cross-check since more letters could throw new candidates into consideration.
        new_cross_check = {letter for letter in LETTER_VALUES if letter != '#'}
        for candidate in list(new_cross_check):
            if candidate not in prefix_node.children:
                new_cross_check.remove(candidate)
                continue
            suffix_sum = 0
            suffix_node = prefix_node.children[candidate]
            lower = r + 1
            while self.board.in_bounds(lower, c) and not self.board[lower][c].vacant():
                sq = self.board[lower][c]
                if sq.value not in suffix_node.children:
                    suffix_node = None
                    break
                suffix_node = suffix_node.children[sq.value]
                suffix_sum += LETTER_VALUES[sq.value]
                lower += 1
            if suffix_node and suffix_node.is_valid_word:
                if not self.board.transposed:
                    self.board[r][c].set_cross_sum_vertical(prefix_sum + suffix_sum)
                else:
                    self.board[r][c].set_cross_sum_horizontal(prefix_sum + suffix_sum)
            else:
                # Unable to proceed to a valid trie node or node is non-terminal so this candidate should be del.
                new_cross_check.remove(candidate)
        if not self.board.transposed:
            self.board[r][c].set_cross_checks_vertical(new_cross_check)
        else:
            self.board[r][c].set_cross_checks_horizontal(new_cross_check)

    def _update_h_cross_check(self, coords: Tuple[int, int]):
        """
        This is somewhat duplicated from _update_v_cross_check with movement across row.
        """
        r, c = coords
        left = c
        while self.board.in_bounds(r, left - 1) and not self.board[r][left - 1].vacant():
            left -= 1
        prefix_node = self.dict_trie
        prefix_sum = 0
        while left < c:
            # TODO: Exception check since prefix should always exist
            sq = self.board[r][left]
            prefix_node = prefix_node.children[sq.value]
            prefix_sum += LETTER_VALUES[sq.value]
            left += 1
        # We must refresh cross-check since more letters could throw new candidates into consideration.
        new_cross_check = {letter for letter in LETTER_VALUES if letter != '#'}
        for candidate in list(new_cross_check):
            if candidate not in prefix_node.children:
                new_cross_check.remove(candidate)
                continue
            suffix_sum = 0
            suffix_node = prefix_node.children[candidate]
            right = c + 1
            while self.board.in_bounds(r, right) and not self.board[r][right].vacant():
                sq = self.board[r][right]
                if sq.value not in suffix_node.children:
                    suffix_node = None
                    break
                suffix_sum += LETTER_VALUES[sq.value]
                suffix_node = suffix_node.children[sq.value]
                right += 1
            if suffix_node and suffix_node.is_valid_word:
                if not self.board.transposed:
                    self.board[r][c].set_cross_sum_horizontal(prefix_sum + suffix_sum)
                else:
                    self.board[r][c].set_cross_sum_vertical(prefix_sum + suffix_sum)
            else:
                # Unable to proceed to a valid trie node or node is non-terminal so this candidate should be del.
                new_cross_check.remove(candidate)
        if not self.board.transposed:
            self.board[r][c].set_cross_checks_horizontal(new_cross_check)
        else:
            self.board[r][c].set_cross_checks_vertical(new_cross_check)

    def apply_best_move(self) -> List[str]:
        res = []
        if not self.best_word or self.best_score <= 0:
            return res
        if self.used_transpose:
            self.board.transpose()
        r, c = self.start_coords
        v_cross_check = []
        for i, ch in enumerate(self.best_word):
            square = self.board[r][c + i]
            if not square.vacant():
                assert square.value == ch, \
                    f"Problem with word, {self.best_word}, formed at {self.start_coords} in {i}th char."
            else:
                # Place a tile
                res.append(ch)
                square.value = ch
            upper = self._find_first_vacant_in_column(r, c + i, functools.partial(add, -1))
            lower = self._find_first_vacant_in_column(r, c + i, functools.partial(add, 1))
            if upper:
                v_cross_check.append(upper)
            if lower:
                v_cross_check.append(lower)
        # Vertical cross-check candidates
        for cc in v_cross_check:
            self._update_v_cross_check(cc)

        # The only horizontal cross-check candidates are on either side of the end of the word
        h_candidates = [(r, c - 1), (r, c + len(self.best_word))]
        for cc in h_candidates:
            if self.board.in_bounds(*cc):
                self._update_h_cross_check(cc)

        # Clear turn state once move is applied
        self._clear_turn_state()

        # Return the used tiles
        return res
