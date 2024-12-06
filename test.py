# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokens = tokenizer.tokenize("cat")
# print(tokens) 

import pickle

# Load the saved embedding dictionary
with open('embedding_dict.pkl', 'rb') as file:
    embedding_dict = pickle.load(file)

# Check the contextual embedding for the word "cat"
if 'cat' in embedding_dict:
    contextual_embedding = embedding_dict['cat']['contextual']
    print(f"Contextual embedding for 'cat':\n{len(contextual_embedding)}")
else:
    print("The word 'cat' is not found in the embedding dictionary.")
