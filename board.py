import functools
import pickle
from dataclasses import dataclass, field
from enum import auto
from operator import add
from typing import Set, Optional, Tuple, List, Callable

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
    cross_checks: Set[str] = field(default_factory=set)
    typ: SquareType = SquareType.NORMAL

    def __post_init__(self):
        self.cross_checks = {letter for letter in LETTER_VALUES if letter != '#'}

    def vacant(self):
        return not self.value

    def __str__(self):
        if self.vacant():
            return str(self.typ)
        return self.value

    def __repr__(self):
        return f"Square({self.typ}, {self.value or str()})"


class Board:
    def __init__(self, lexicon_file='lexicon/scrabble_word_list.pickle'):
        self.board = [[Square() for i in range(15)] for j in range(15)]
        self.add_premium_squares()
        self.words_played = set()

        with open(lexicon_file, 'rb') as f:
            self.dict_trie: TrieNode = pickle.load(f)

        # Turn state
        self.best_word = ""
        self.start_coords = (-1, -1)
        self.best_score = 0

    @staticmethod
    def _in_bounds(r, c):
        return 0 <= r < 15 and 0 <= c < 15

    def _is_potential_anchor(self, r, c):
        adjacent_squares = [self.board[r2][c2] for r2, c2 in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if
                            self._in_bounds(r2, c2)]
        # At least one of the adjacent squares must have a letter to tether to
        return self.board[r][c].vacant() and any(not square.vacant() for square in adjacent_squares)

    def score_word(self, word: str, coords: Tuple[int, int]):
        r, c = coords
        score = 0
        word_multiplier = 1
        new_letters = 0
        for i, ch in enumerate(word):
            square = self.board[r][c + i]
            if not square.vacant():
                assert square.value == ch, f"Problem with word, {word}, formed at {coords} in {i}th char."
                score += LETTER_VALUES[ch]
            else:
                mult = square.typ.letter_multiplier()
                score += (LETTER_VALUES[ch] * mult)
                word_multiplier *= square.typ.word_multiplier()
                new_letters += 1
        # TODO: does not yet account for words formed vertically
        score *= word_multiplier
        if new_letters == 7:
            score += 50
        if score > self.best_score:
            self.best_word = word
            self.start_coords = coords
            self.best_score = score
        return score

    def extend_right(
            self,
            partial_word: str,
            trie_node: TrieNode, tiles: List[str],
            coords: Tuple[int, int],
            anchor_placed: bool = True
    ):
        r, c = coords
        if not self._in_bounds(r, c) or not tiles:
            # Since we are out of bounds / no tiles, no further exploration can happen
            if trie_node.is_valid_word and anchor_placed:
                self.score_word(partial_word, (r, c - len(partial_word)))
            return
        square = self.board[r][c]
        if square.vacant:
            if trie_node.is_valid_word and anchor_placed:
                self.score_word(partial_word, (r, c - len(partial_word)))
                pass
            for letter, child_node in trie_node.children.items():
                if letter not in tiles or letter not in square.cross_checks:
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

    def find_best_move(self, tiles: List[str]):
        self.best_score = 0
        self.best_word = ""
        self.start_coords = (-1, -1)
        if self.board[7][7].vacant():
            # start of game
            self.left_part("", self.dict_trie, tiles, 6, (7, 7))
            return
        # middle of game, solve using anchors
        for r in range(15):
            # We solve the best for each row
            # We must keep track of the number of vacant / unconnected squares left of anchor
            left_of_anchor = 0
            prefix = ""
            for c in range(15):
                if not self.board[r][c].vacant():
                    prefix += self.board[r][c].value
                    left_of_anchor = 0
                    continue
                if not self._is_potential_anchor(r, c):
                    # vacant non-anchor -> increment left and clear prefix
                    prefix = ""
                    left_of_anchor += 1
                    continue
                # Found an anchor, run alg and reset left counter
                start_node = self.dict_trie.traverse_prefix(prefix)
                if start_node:
                    self.left_part(prefix, start_node, tiles, left_of_anchor, (r, c))
                left_of_anchor = 0

    def _clear_turn_state(self):
        self.best_score = 0
        self.best_word = ""
        self.start_coords = (-1, -1)

    def _find_first_vacant_in_column(self, r: int, c: int, direction: Callable):
        r = direction(r)
        while self._in_bounds(r, c) and not self.board[r][c].vacant():
            r = direction(r)
        if self._in_bounds(r, c):
            return r, c
        return None

    def _update_cross_check(self, coords: Tuple[int, int]):
        """
        This method can be broken down into 2 phases.
        1. Track up and find the prefix above the cross-check coords.
        2. Find whether any current letter in existing cross-check leads to a valid word
        with the computed prefix (above) and computed suffix (walking down column).
        """
        r, c = coords
        upper = r
        while self._in_bounds(upper - 1, c) and not self.board[upper - 1][c].vacant():
            upper -= 1
        prefix_node = self.dict_trie
        while upper < r:
            # TODO: Exception check since prefix should always exist
            prefix_node = prefix_node.children[self.board[upper][c].value]
            upper += 1
        candidates = set()
        for candidate in self.board[r][c].cross_checks:
            if candidate not in prefix_node.children:
                continue
            suffix_node = prefix_node.children[candidate]
            lower = r+1
            while self._in_bounds(lower, c) and not self.board[lower][c].vacant():
                sq = self.board[lower][c]
                if sq.value not in suffix_node.children:
                    suffix_node = None
                    break
                suffix_node = suffix_node.children[sq.value]
                lower += 1
            if suffix_node and suffix_node.is_valid_word:
                candidates.add(candidate)
        self.board[r][c].cross_checks = candidates

    def apply_best_move(self):
        # TODO: compute cross-checks on each successful move
        if not self.best_word or self.best_score <= 0:
            return
        r, c = self.start_coords
        to_cross_check = []
        for i, ch in enumerate(self.best_word):
            square = self.board[r][c + i]
            if not square.vacant():
                assert square.value == ch, \
                    f"Problem with word, {self.best_word}, formed at {self.start_coords} in {i}th char."
            else:
                square.value = ch
            upper = self._find_first_vacant_in_column(r, c + i, functools.partial(add, -1))
            lower = self._find_first_vacant_in_column(r, c + i, functools.partial(add, 1))
            if upper:
                to_cross_check.append(upper)
            if lower:
                to_cross_check.append(lower)
        # Cross-check all candidates
        for cc in to_cross_check:
            self._update_cross_check(cc)
        self._clear_turn_state()

    def add_premium_squares(self):
        # Adds all the premium squares that influence the word's score.
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
