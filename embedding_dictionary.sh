#!/bin/bash

# Prompt for the sentence input
echo "Enter your sentence:"
read sentence

# Run the Python script
python3 embedding_dictionary.py --sentence "$sentence"
