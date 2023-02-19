import pickle
from dataclasses import dataclass, field
from typing import Dict, Optional

LEXICON_TXT_FILE = 'lexicon/scrabble_word_list.txt'
PICKLE_FILE = 'lexicon/scrabble_word_list.pickle'


@dataclass
class TrieNode:
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    is_valid_word: bool = False

    def mark_terminal(self):
        self.is_valid_word = True

    def get_or_create_child(self, ch: str) -> 'TrieNode':
        if ch not in self.children:
            self.children[ch] = TrieNode()
        return self.children[ch]

    def traverse_prefix(self, prefix: str) -> Optional['TrieNode']:
        node = self
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    @staticmethod
    def construct_trie(filename: str) -> 'TrieNode':
        root = TrieNode()
        with open(filename) as f:
            for line in f:
                word = line.rstrip().upper()
                node = root
                for ch in word:
                    node = node.get_or_create_child(ch)
                node.mark_terminal()
        return root

    def count_nodes(self):
        if not self.children:
            return 1
        return 1 + sum(child.count_nodes() for _, child in self.children.items())


def construct_scrabble_lexicon(target: str = PICKLE_FILE):
    """
    To check correctness, add this:

    print(trie.count_nodes())
    with open(target, 'rb') as f:
        print(pickle.load(f).count_nodes())

    """
    trie = TrieNode.construct_trie(LEXICON_TXT_FILE)
    with open(target, 'wb') as f:
        pickle.dump(trie, f)


if __name__ == '__main__':
    construct_scrabble_lexicon()

