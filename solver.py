import functools
import multiprocessing
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
        self.cross_sum_vertical = val

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
        # TODO: store these in defaultdict(int) with the associated multipliers for word/letter - will be cleaner
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

    def fill_scenario(self, scenario: List[List[str]]):
        assert len(scenario) == self.board.n and len(scenario[0]) == self.board.n, \
            f"Wrong size board ({len(scenario)}, {len(scenario[0])})."
        for r in range(self.board.n):
            for c in range(self.board.n):
                if scenario[r][c] != '_':
                    self.board[r][c].value = scenario[r][c]
        # For all vacant spaces adjacent to some filled space (ie, anchor), compute cross-checks
        anchors = [(r, c) for r in range(self.board.n) for c in range(self.board.n) if self._is_potential_anchor(r, c)]
        for anchor in anchors:
            self._compute_and_set_cross_check(anchor)

    def _validate_horizontal(self):
        for r in range(self.board.n):
            word_len = 0
            node = self.dict_trie
            for c in range(self.board.n):
                if not self.board[r][c].vacant():
                    assert self.board[r][c].value in node.children, f"Could not traverse trie at {r},{c}"
                    node = node.children[self.board[r][c].value]
                    word_len += 1
                else:
                    # When we find a vacant space, we check whether our existing that the node is valid IF word > 1
                    assert word_len <= 1 or node.is_valid_word, f"Found unknown {word_len}-length word ending {r},{c}."
                    word_len = 0
                    node = self.dict_trie
            # Check word at end of row
            assert word_len <= 1 or node.is_valid_word, \
                f"Found unknown {word_len}-length word ending {r},{self.board.n - 1}."

    def validate_board(self):
        assert not self.board.transposed, "This operation requires the board to not be transposed."
        print("Validating horizontal...")
        self._validate_horizontal()
        self.board.transpose()
        print("Validating vertical...")
        self._validate_horizontal()
        self.board.transpose()
        print("Successful validation!")

    def _is_potential_anchor(self, r, c):
        """
        Every move (except the very first) requires an anchor, a cell which is currently vacant but adjacent to
        at least one occupied cell.  We build potential moves around anchor cells.
        """
        adjacent_squares = [self.board[r2][c2] for r2, c2 in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if
                            self.board.in_bounds(r2, c2)]
        # At least one of the adjacent squares must have a letter to tether to
        return self.board[r][c].vacant() and any(not square.vacant() for square in adjacent_squares)

    def score_word(self, word: str, coords: Tuple[int, int]) -> Tuple[int, Tuple[int, int], str, bool, int]:
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
        return score, coords, word, self.board.transposed, new_letters

    def extend_right(
            self,
            partial_word: str,
            trie_node: TrieNode, tiles: List[str],
            coords: Tuple[int, int],
            anchor_placed: bool = True
    ) -> List[Tuple[int, Tuple[int, int], str, bool, int]]:
        r, c = coords
        res = []
        if not self.board.in_bounds(r, c):
            # Since we are out of bounds, no further exploration can happen
            # Previously, we checked if we had no tiles here but we MUST continue until end of board or vacancy
            if trie_node.is_valid_word and anchor_placed:
                res.append(
                    self.score_word(
                        partial_word,
                        (r, c - len(partial_word))
                    )
                )
            return res
        square = self.board[r][c]
        if square.vacant():
            if trie_node.is_valid_word and anchor_placed:
                res.append(
                    self.score_word(
                        partial_word,
                        (r, c - len(partial_word))
                    )
                )
            for letter, child_node in trie_node.children.items():
                # We always form words horizontally but change the orientation of the board repr beforehand.
                cross_checks = square.cross_checks_vertical \
                    if not self.board.transposed else square.cross_checks_horizontal
                if letter not in tiles or letter not in cross_checks:
                    continue
                tiles.remove(letter)
                res.extend(
                    self.extend_right(partial_word + letter, child_node, tiles, (r, c + 1))
                )
                tiles.append(letter)
        else:
            # square already occupied by letter
            letter = square.value
            if letter in trie_node.children:
                child_node = trie_node.children[letter]
                res.extend(
                    self.extend_right(partial_word + letter, child_node, tiles, (r, c + 1))
                )
        return res

    def left_part(
            self,
            partial_word: str,
            trie_node: TrieNode,
            tiles: List[str],
            limit: int,
            anchor: Tuple[int, int]
    ) -> List[Tuple[int, Tuple[int, int], str, bool, int]]:
        # The left part is guaranteed to have no cross-checks, so we follow trie based on tiles
        res = []
        res.extend(self.extend_right(partial_word, trie_node, tiles, anchor, anchor_placed=False))
        if limit > 0:
            for letter, child_node in trie_node.children.items():
                if letter not in tiles:
                    continue
                tiles.remove(letter)
                res.extend(self.left_part(partial_word + letter, child_node, tiles, limit - 1, anchor))
                tiles.append(letter)
        return res

    def _best_move_helper(self, tiles: List[str]) -> List[Tuple[int, Tuple[int, int], str, bool, int]]:
        # When in the middle of game, solve using anchors
        results = []
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
                # Start node should always exist since the prefix exists on the board.
                if start_node:
                    results.extend(self.left_part(prefix, start_node, tiles, left_of_anchor, (r, c)))
                # Since this anchor was both vacant and connected, prefix and left must be reset.
                prefix = ""
                left_of_anchor = 0
        """
        This actually does NOT help surprisingly... it was already fast to begin with and CPU-bound I guess?

        args = []
        args.append((prefix, start_node, tiles, left_of_anchor, (r, c)))
        with multiprocessing.Pool(8) as pool:
            for inner_results in pool.starmap(self.left_part, args):
                self._find_best_result(inner_results)
        """
        return results

    def find_best_move(self, tiles: List[str]):
        self._clear_turn_state()
        results = []
        if self.board[7][7].vacant():
            # start of game, apply best move horizontally since symmetric
            results.extend(self.left_part("", self.dict_trie, tiles, 6, (7, 7)))
        else:
            results.extend(self._best_move_helper(tiles))
            # Solve for vertically formed words using transpose
            self.board.transpose()
            results.extend(self._best_move_helper(tiles))
            # Undo transpose
            self.board.transpose()
        print(f"Found {len(results)} valid words during search.")
        self._find_best_result(results)

    def _find_best_result(self, results):
        # Pick word that gives most points per letter placed
        for score, coords, word, trans, new_letters in results:
            avg_score = score / new_letters
            if avg_score > self.score_per_letter:
                self.best_word = word
                self.best_score = score
                self.start_coords = coords
                self.used_transpose = trans
                self.score_per_letter = avg_score
        return

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

    def _compute_cross_check_vertical(self, coords: Tuple[int, int]):
        """
        This re-computes the vertical cross-check and cross-sum for a given coordinate.
        The steps to be taken:
        1. Find the prefix above the coordinates and the suffix below the coordinates and the total point value.
        1a. If both are empty, the cross-check for this should be the full letter-set, c-sum 0, and we early-return.
        1b. If not, your cross sum is the total point value computed for the prefix and suffix.
        2. From the prefix node, try all letter candidates and see if they lead to a valid word with the suffix.
        Explained better: Find whether any letter leads to a valid word
        with the computed prefix (above) and computed suffix (walking down column).
        3. For the candidates that don't, remove from the running set for new cross-check.
        """
        prefix = ""
        suffix = ""
        new_cross_check = {letter for letter in LETTER_VALUES if letter != '#'}
        r, c = coords
        # Find prefix
        upper = r
        cross_sum = 0
        while self.board.in_bounds(upper - 1, c) and not self.board[upper - 1][c].vacant():
            upper -= 1
            sq = self.board[upper][c]
            prefix = sq.value + prefix
            cross_sum += LETTER_VALUES[sq.value]
        prefix_node = self.dict_trie.traverse_prefix(prefix)
        # Find suffix
        lower = r + 1
        while self.board.in_bounds(lower, c) and not self.board[lower][c].vacant():
            sq = self.board[lower][c]
            suffix += sq.value
            cross_sum += LETTER_VALUES[sq.value]
            lower += 1
        if not prefix and not suffix:
            # No cross-check needed - for good measure, set cross-check to full set and cross-sum to 0.
            return new_cross_check, 0
        # Compute valid candidates
        for candidate in list(new_cross_check):
            node = prefix_node
            for suffix_letter in candidate + suffix:
                if suffix_letter not in node.children:
                    node = None
                    break
                node = node.children[suffix_letter]
            if not node or not node.is_valid_word:
                # Unable to track down trie for suffix / not at a valid leaf -> eliminate candidate
                new_cross_check.remove(candidate)
        return new_cross_check, cross_sum

    def _compute_and_set_cross_check(self, coords: Tuple[int, int]):
        assert not self.board.transposed, "This operation requires the board to not be transposed."
        r, c = coords
        v_check, v_sum = self._compute_cross_check_vertical(coords)
        self.board.transpose()
        # Compute horizontal as "vertical" post-transpose with flipped dimensions
        h_check, h_sum = self._compute_cross_check_vertical((c, r))
        self.board.transpose()
        # Set vertical and horizontal checks as computed above
        self.board[r][c].set_cross_sum_vertical(v_sum)
        self.board[r][c].set_cross_checks_vertical(v_check)
        self.board[r][c].set_cross_sum_horizontal(h_sum)
        self.board[r][c].set_cross_checks_horizontal(h_check)

    def apply_best_move(self) -> List[str]:
        assert not self.board.transposed, "This operation requires the board to not be transposed."
        res = []
        if not self.best_word or self.best_score <= 0:
            return res
        if self.used_transpose:
            self.board.transpose()
        r, c = self.start_coords
        # The max size of all_candidates is 2 * num_letters_placed + 2 <= 16.
        all_candidates = []
        for i, ch in enumerate(self.best_word):
            square = self.board[r][c + i]
            if not square.vacant():
                # The tile was placed previously - we need not update any vertical associated with this column.
                assert square.value == ch, \
                    f"Problem with word, {self.best_word}, formed at {self.start_coords} in {i}th char."
                continue
            # Place a tile
            res.append(ch)
            square.value = ch
            upper = self._find_first_vacant_in_column(r, c + i, functools.partial(add, -1))
            lower = self._find_first_vacant_in_column(r, c + i, functools.partial(add, 1))
            if upper:
                all_candidates.append(upper)
            if lower:
                all_candidates.append(lower)

        # The only horizontal cross-check candidates are on either side of the end of the word
        h_candidates = [(r, c - 1), (r, c + len(self.best_word))]
        all_candidates.extend([(rp, cp) for rp, cp in h_candidates if self.board.in_bounds(rp, cp)])

        # Flip candidates dimensions and clear turn state once move is applied
        if self.board.transposed:
            all_candidates = [(c, r) for r, c in all_candidates]
        self._clear_turn_state()

        for candidate in all_candidates:
            self._compute_and_set_cross_check(candidate)

        # Return the used tiles
        return res
