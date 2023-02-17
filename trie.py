import pickle
from dataclasses import dataclass, field
from typing import Dict, Optional


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


if __name__ == '__main__':
    file = 'lexicon/scrabble_word_list.txt'
    trie = TrieNode.construct_trie(file)
    print(trie.count_nodes())
    pickle_file = 'lexicon/scrabble_word_list.pickle'
    with open(pickle_file, 'wb') as f:
        pickle.dump(trie, f)
    with open(pickle_file, 'rb') as f:
        print(pickle.load(f).count_nodes())

