from itertools import permutations
from collections import defaultdict
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np

def build_tree(sentence):
    words = sentence.split()
    tree = defaultdict(set)
    
    # Generate permutations for each level
    for level in range(1, len(words) + 1):
        for perm in permutations(words, level):
            phrase = ' '.join(perm)
            tree[level].add(phrase)  # Add unique phrases to each level
    
    return tree

def load_word2vec_model():
    # Load a pre-trained Word2Vec model (e.g., Google News vectors)
    # Replace 'path/to/word2vec/model' with the actual path to your model
    model = api.load("word2vec-google-news-300")
    return model

def phrase_to_embedding(phrase, model):
    words = phrase.split()
    embeddings = []
    
    # Gather embeddings for words in the phrase
    for word in words:
        if word in model:
            embeddings.append(model[word])
    
    # Compute the average embedding for the phrase
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return None  # Return None if no words are in the model

def tree_to_embeddings(tree, model):
    tree_embeddings = defaultdict(list)
    
    for level, phrases in tree.items():
        for phrase in phrases:
            embedding = phrase_to_embedding(phrase, model)
            if embedding is not None:
                tree_embeddings[level].append((phrase, embedding))
    
    return tree_embeddings

# Example usage
sentence = "this is a cat"
word_tree = build_tree(sentence)
model = load_word2vec_model()
word_tree_embeddings = tree_to_embeddings(word_tree, model)

# Print embeddings for each level
for level, phrases in word_tree_embeddings.items():
    print(f"Level {level}:")
    for phrase, embedding in phrases:
        print(f"  Phrase: '{phrase}'")
        print(f"  Embedding: {embedding[:5]}...")  # Print the first 5 dimensions of embedding for brevity
