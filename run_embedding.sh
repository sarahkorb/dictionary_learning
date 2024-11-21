#!/bin/bash

# Prompt for input type
echo "Do you want to input a sentence or provide a text file? (Type '1' for sentence or '2' for text file)"
read input_choice

if [ "$input_choice" == "1" ]; then
    input_type="sentence"
    echo "Enter your sentence:"
    read sentence
    input_data="$sentence"
elif [ "$input_choice" == "2" ]; then
    input_type="txt"
    echo "Enter the path to your .txt file:"
    read txt_file
    if [ ! -f "$txt_file" ]; then
        echo "File not found! Exiting."
        exit 1
    fi
    # Check if the file has a .txt extension
    if [[ "$txt_file" != *.txt ]]; then
        echo "The file must be a .txt file! Exiting."
        exit 1
    fi
    input_data="$txt_file"
else
    echo "Invalid input choice! Exiting."
    exit 1
fi

# Prompt for embedding model
echo "Which encoder do you want to use? (Type '1' for Word2Vec or '2' for BERT)"
read encoder_choice

if [ "$encoder_choice" == "1" ]; then
    encoder="word2vec"
elif [ "$encoder_choice" == "2" ]; then
    encoder="bert"
else
    echo "Invalid encoder choice! Exiting."
    exit 1
fi

# Prompt for permute option (only available for sentence input)
if [ "$input_type" == "sentence" ]; then
    echo "Do you want to permute the sentence? (Type 'yes' for permutation or 'no' to keep the original order)"
    read permute_choice
else
    # Disable permute choice for text file input
    permute_choice="no"
fi

# Prompt for embedding choice (word-level or contextual)
echo "Do you want word-level embeddings or contextual embeddings? (Type 'word' for word-level or 'contextual' for contextual)"
read embedding_choice

# Run the Python script with the necessary arguments
python3 embedding_script.py --input_type "$input_type" --input_data "$input_data" --encoder "$encoder" --permute "$permute_choice" --embedding_choice "$embedding_choice"

# Inform the user about the output
if [ "$input_type" == "sentence" ]; then
    echo "Embeddings for the sentence have been saved to 'sentence_embeddings.csv'."
else
    echo "Embeddings for the text file have been saved to 'txt_embeddings.csv'."
fi
