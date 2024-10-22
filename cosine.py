import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Load your pretrained dictionary from the .pt file
pretrained_dict = torch.load('dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768/ae.pt', map_location=torch.device('cpu'),weights_only=True)

# Access the encoder weights
encoder_weights = pretrained_dict['encoder.weight']  # Use encoder weights for cosine similarity

print(f"Encoder weights shape: {encoder_weights.shape}")
# 32768 rows: latent features
# 512 columns: Each feature vector is in 512-dimensional space

# Step 1: Normalize each row (feature) to unit length
norms = encoder_weights.norm(dim=1, keepdim=True)  # Compute L2 norm of each row
normalized_weights = encoder_weights / norms  # Normalize rows to unit length

# Step 2: Compute the cosine similarity matrix
#Since we normalised, this is just dot product 
cosine_similarities = torch.matmul(normalized_weights, normalized_weights.T)

# Step 3: Print the result
print("Cosine Similarities:\n", cosine_similarities)

# Step 1: Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(normalized_weights.cpu().numpy())

# Step 2: Scatter plot the reduced vectors
plt.figure(figsize=(10, 8))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5)
plt.title("Feature Vectors Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# # Step 2: Plot heatmap (show a subset for readability, e.g., top 500x500)
# subset_size = 500  # Adjust this to avoid overwhelming visualization
# cosine_subset = cosine_similarities[:subset_size, :subset_size]

# plt.figure(figsize=(10, 8))
# sns.heatmap(cosine_subset.numpy(), cmap="coolwarm", center=0)
# plt.title("Cosine Similarities Heatmap (Subset)")
# plt.show()
