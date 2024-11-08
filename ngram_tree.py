from itertools import permutations
from collections import defaultdict

def Ngram_tree(sentence):
    words = sentence.split()
    tree = defaultdict(list)
    
    for level in range(1, len(words) + 1):
        # Generate permutations of a certain length (length represented by level)
        for perm in permutations(words, level):
            phrase = ' '.join(perm)
            tree[level].append(phrase)
    
    return tree

def print_tree(tree):
    for level, phrases in tree.items():
        print(f"Level {level}:")
        for phrase in sorted(phrases):
            print(f"  {'  ' * (level - 1)}{phrase}")  # Indent according to level

#EXAMPLE:
sentence = "this is a cat"
word_tree = Ngram_tree(sentence)
print_tree(word_tree)

