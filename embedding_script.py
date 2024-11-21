import argparse
import pandas as pd
from itertools import permutations
from collections import defaultdict
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import pandas as pd

# BERT imports (install transformers library if needed)
from transformers import AutoTokenizer, AutoModel
import torch

def build_tree(sentence):
    words = sentence.split()
    tree = defaultdict(set)
    for level in range(1, len(words) + 1):
        for perm in permutations(words, level):
            phrase = ' '.join(perm)
            tree[level].add(phrase)
    return tree

def load_word2vec_model():
    return api.load("word2vec-google-news-300")

def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

def phrase_to_embedding_word2vec(phrase, model):
    words = phrase.split()
    embeddings = [model[word] for word in words if word in model]
    return np.mean(embeddings, axis=0) if embeddings else None

def phrase_to_embedding_bert(phrase, tokenizer, model):
    # Tokenize the input phrase
    inputs = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True) 
    
    # Get the output embeddings from the BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings for each token
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (num_tokens, 768)  
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))  # Decode token IDs
    
    # Keep only embeddings for original words
    word_embeddings = {}
    word_list = phrase.split()  # Split input phrase into words
    
    token_idx = 1  # Start from 1 to skip the [CLS] token
    for word in word_list:
        word_embedding = []
        # Go through the tokens for each word, skip [CLS] and [SEP] tokens
        while token_idx < len(tokens) and tokens[token_idx] != '[SEP]' and tokens[token_idx] != '[CLS]' and word.lower() in tokens[token_idx].lower():
            word_embedding.append(token_embeddings[token_idx].tolist())  # Append the embedding for each token of the word
            token_idx += 1
        if word_embedding:
            # Store the average of the word's embeddings as the final embedding for the word
            word_embeddings[word] = np.mean(word_embedding, axis=0)   
    
    return word_embeddings


def tree_to_embeddings(tree, encoder, model, tokenizer, embedding_choice):
    tree_embeddings = []
    for level, phrases in tree.items():
        for phrase in phrases:
            if encoder == "bert":
                if embedding_choice == "word":
                    embedding_dict = phrase_to_embedding_bert(phrase, tokenizer, model)
                    for word, embedding in embedding_dict.items():
                        tree_embeddings.append({"level": level, "phrase": phrase, "word": word, "embedding": embedding})
                elif embedding_choice == "contextual":
                    embedding_dict = phrase_to_embedding_bert(phrase, tokenizer, model)
                    tree_embeddings.append({"level": level, "phrase": phrase, "embedding": np.mean(list(embedding_dict.values()), axis=0)})
            else:  # word2vec
                embedding = phrase_to_embedding_word2vec(phrase, model)
                tree_embeddings.append({"level": level, "phrase": phrase, "embedding": embedding.tolist()})
    
    return tree_embeddings

def process_sentence(sentence, encoder, model, tokenizer=None, permute_choice="no", embedding_choice="word"):
    if permute_choice == "yes":
        tree = build_tree(sentence)
        return tree_to_embeddings(tree, encoder, model, tokenizer, embedding_choice)
    else:
        # Directly generate embeddings for the original sentence
        if encoder == "bert":
            if embedding_choice == "word":
                embedding_dict = phrase_to_embedding_bert(sentence, tokenizer, model)
                return [{"level": 1, "phrase": sentence, "word": word, "embedding": embedding} for word, embedding in embedding_dict.items()]
            elif embedding_choice == "contextual":
                embedding_dict = phrase_to_embedding_bert(sentence, tokenizer, model)
                return [{"level": 1, "phrase": sentence, "embedding": np.mean(list(embedding_dict.values()), axis=0)}]
        else:  # word2vec
            embedding = phrase_to_embedding_word2vec(sentence, model)
            return [{"level": 1, "phrase": sentence, "embedding": embedding.tolist()}]

def process_txt(file_path, encoder, model, tokenizer=None, embedding_choice="word"):
    with open(file_path, "r") as file:
        sentences = file.readlines()
    
    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        embeddings = process_sentence(sentence, encoder, model, tokenizer, permute_choice="no", embedding_choice=embedding_choice)
        results.extend(embeddings)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", choices=["sentence", "txt"], required=True)
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--encoder", choices=["word2vec", "bert"], required=True)
    parser.add_argument("--permute", choices=["yes", "no"], default="no", help="Whether to permute the input.")
    parser.add_argument("--embedding_choice", choices=["word", "contextual"], default="word", help="Word-level or contextual embeddings.")
    args = parser.parse_args()

    # Disable the permute option for text file input
    if args.input_type == "txt" and args.permute == "yes":
        print("Permutation is not allowed for text file inputs. Setting permute to 'no'.")
        args.permute = "no"

    if args.encoder == "word2vec":
        model = load_word2vec_model()
        tokenizer = None
    elif args.encoder == "bert":
        tokenizer, model = load_bert_model()

    if args.input_type == "sentence":
        embeddings = process_sentence(args.input_data, args.encoder, model, tokenizer, args.permute, args.embedding_choice)
        output_file = "sentence_embeddings.csv"
    elif args.input_type == "txt":
        embeddings = process_txt(args.input_data, args.encoder, model, tokenizer, args.embedding_choice)
        output_file = "txt_embeddings.csv"

    # Save embeddings to CSV
    df = pd.DataFrame(embeddings)
    if not df.empty:
        print(f"Saving {len(df)} embeddings to {output_file}...")
        df.to_csv(output_file, index=False)
    else:
        print("No embeddings to save.")
