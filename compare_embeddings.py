import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embedding_dictionary import load_word2vec_model, load_bert_model, get_contextual_embeddings, get_word2vec_embeddings

# Compute cosine similarity between two embeddings
def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Create and compare embeddings
def create_and_compare_embeddings(sentence1, sentence2, tokenizer, model, word2vec_model):
    bert_embeddings_1 = get_contextual_embeddings(sentence1, tokenizer, model)
    word2vec_embeddings_1 = get_word2vec_embeddings(sentence1, word2vec_model)
    
    bert_embeddings_2 = get_contextual_embeddings(sentence2, tokenizer, model)
    word2vec_embeddings_2 = get_word2vec_embeddings(sentence2, word2vec_model)

    # Cosine similarities for contextual vs non-contextual embeddings
    similarity_dict = {'contextual': [], 'non_contextual': []}
    
    # Get pairwise cosine similarity for each word
    for word1 in sentence1.split():
        if word1 in bert_embeddings_1 and word1 in bert_embeddings_2:
            contextual_sim = compute_cosine_similarity(bert_embeddings_1[word1], bert_embeddings_2[word1])
            similarity_dict['contextual'].append((word1, contextual_sim))
        
        if word1 in word2vec_embeddings_1 and word1 in word2vec_embeddings_2:
            non_contextual_sim = compute_cosine_similarity(word2vec_embeddings_1[word1], word2vec_embeddings_2[word1])
            similarity_dict['non_contextual'].append((word1, non_contextual_sim))

    return similarity_dict

# Main entry point
if __name__ == "__main__":
    sentence1 = input("Enter the first sentence: ")
    sentence2 = input("Enter the second sentence: ")

    # Load models
    word2vec_model = load_word2vec_model()
    bert_tokenizer, bert_model = load_bert_model()

    # Create embeddings and compute similarity
    similarity_dict = create_and_compare_embeddings(sentence1, sentence2, bert_tokenizer, bert_model, word2vec_model)

    print("Pairwise Cosine Similarities:")
    for embedding_type in similarity_dict:
        print(f"\n{embedding_type.capitalize()} Cosine Similarities:")
        for word, sim in similarity_dict[embedding_type]:
            print(f"Word: '{word}', Similarity: {sim:.4f}")
