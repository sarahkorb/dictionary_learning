import argparse
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import torch
import numpy as np
import pickle

def load_word2vec_model():
    model = api.load("word2vec-google-news-300")
    print(f"Loaded Word2Vec model: {model}")  # Debugging: check if model is loaded correctly
    return model

def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

def get_contextual_embeddings(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (num_tokens, 768)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

    print(f"Tokens: {tokens}")  # Debugging: check how BERT tokenizes the input
    word_embeddings = {}
    
    for word in sentence.split():
        word_embedding = []
        word_tokens = tokenizer.tokenize(word)  # Get subwords for the word
        
        # Check if the word is split into subwords
        if len(word_tokens) > 1:
            print(f"Word '{word}' is split into subwords: {word_tokens}")
        
        is_split = False
        token_idx = 1  # Skip [CLS]
        
        while token_idx < len(tokens) and tokens[token_idx] != "[SEP]":
            token = tokens[token_idx].lower()
            # Check if the token matches the word or subword
            if any(subword.lower() in token for subword in word_tokens):
                word_embedding.append(token_embeddings[token_idx].tolist())
                if len(word_tokens) > 1:
                    is_split = True  # Mark if the word is split into subwords
            token_idx += 1

        if word_embedding:
            if is_split:
                word_embeddings[word] = np.mean(word_embedding, axis=0)  # Combine subword embeddings
            else:
                # If the word wasn't split, use the embedding directly for the word
                word_embeddings[word] = word_embedding[0]  # Only one token's embedding
        else:
            print(f"Warning: '{word}' not found in BERT model, skipping...")
    
    return word_embeddings


def get_word2vec_embeddings(sentence, model):
    word_list = sentence.split()
    word_embeddings = {}
    for word in word_list:
        if word in model:
            word_embeddings[word] = model[word]
        else:
            print(f"Warning: '{word}' not found in Word2Vec model, assigning zero vector...")
            word_embeddings[word] = np.zeros(300)  # Word2Vec has 300-dimensional embeddings
    return word_embeddings

def create_embedding_dictionary(sentence, bert_tokenizer, bert_model, word2vec_model):
    bert_embeddings = get_contextual_embeddings(sentence, bert_tokenizer, bert_model)
    word2vec_embeddings = get_word2vec_embeddings(sentence, word2vec_model)
    
    embedding_dict = {}
    for word in sentence.split():
        embedding_dict[word] = {
            "contextual": bert_embeddings.get(word, None),
            "non_contextual": word2vec_embeddings.get(word, None),
        }
    return embedding_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", required=True, help="Input sentence.")
    args = parser.parse_args()

    # Load models
    word2vec_model = load_word2vec_model()
    bert_tokenizer, bert_model = load_bert_model()

    # Create the dictionary
    embedding_dict = create_embedding_dictionary(args.sentence, bert_tokenizer, bert_model, word2vec_model)

    with open("embedding_dict.pkl", "wb") as file:
        pickle.dump(embedding_dict, file)

    print("Dictionary saved to 'embedding_dict.pkl'")
    # print(embedding_dict)
