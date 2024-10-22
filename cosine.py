import torch

# Load your pretrained dictionary from the .pt file
pretrained_dict = torch.load('dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768/ae.pt', map_location=torch.device('cpu'),weights_only=True)

# Access the encoder weights
encoder_weights = pretrained_dict['encoder.weight']  # Use encoder weights for cosine similarity

# Step 1: Define the cosine similarity function
def cosine_similarity(a, b):
    """Calculate cosine similarity between two tensors."""
    cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)
    return cos_sim

# Step 2: Calculate cosine similarity for the encoder weights
similarities = cosine_similarity(encoder_weights, encoder_weights)

# Output the cosine similarity results
print("Cosine Similarities of Encoder Weights:\n", similarities)


