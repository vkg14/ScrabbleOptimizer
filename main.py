from solver import Solver
from trie import construct_scrabble_lexicon
from tilebag import TileBag
from tabulate import tabulate


def print_racks(rack1, rack2):
    print(rack1)
    print(rack2)


def explore_error_case(filename: str):
    with open(filename) as f:
        content = f.read()
    board, rack, reason = content.split('\n\n')
    b = []
    for row in board.split('\n'):
        r = row.split()
        b.append(r)
    print(reason)

    construct_scrabble_lexicon()
    s = Solver()
    s.fill_scenario(b)
    print(tabulate(s.board.board))
    tiles = [ch for ch in rack.strip()]
    s.find_best_move(tiles)
    print(f'{s.best_word}, {s.best_score}')


def sample_game():
    bag = TileBag()
    rack1 = [bag.take_from_bag().get_letter() for _ in range(7)]
    rack2 = [bag.take_from_bag().get_letter() for _ in range(7)]
    print_racks(rack1, rack2)

    # Construct trie and solver
    construct_scrabble_lexicon()
    s = Solver()

    turns = 0
    score = 0
    while True:
        rack = rack1 if turns % 2 == 0 else rack2
        tiles_left = bag.get_remaining_tiles()
        s.find_best_move(rack)
        if s.best_score > 0:
            inverted_s = " with rotation" if s.used_transpose else ""
            print(f'Found move {s.best_word} with value {s.best_score} at {s.start_coords}{inverted_s}.')

            # Increment stats
            score += s.best_score
            turns += 1

            # Apply move and replenish tiles
            used_tiles = s.apply_best_move()
            for t in used_tiles:
                rack.remove(t)
            for _ in range(min(len(used_tiles), tiles_left)):
                rack.append(bag.take_from_bag().get_letter())
        else:
            # best_score == 0
            if tiles_left == 0:
                break
            else:
                print(f'Swapping tiles since no valid move was found.')
                new_tiles = bag.swap_tiles(rack)
                rack.clear()
                rack.extend(new_tiles)

        # Print board and rack state
        print(tabulate(s.board.board))
        print_racks(rack1, rack2)
        print(f'{bag.get_remaining_tiles()} tiles left in the bag.')

        # If there are no more tiles in our rack, end the game.
        if not rack:
            break

    print(f'Game finished in {turns} turns with total score of {score}.')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # explore_error_case('error_cases/inappropriate_connection.txt')
    sample_game()
