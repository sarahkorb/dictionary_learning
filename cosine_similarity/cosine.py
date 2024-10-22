import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

# Load your pretrained dictionary from the .pt file
pretrained_dict = torch.load('dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768/ae.pt', map_location=torch.device('cpu'),weights_only=True)

# Access the encoder weights
encoder_weights = pretrained_dict['encoder.weight']  # Use encoder weights for cosine similarity

print(f"Encoder weights shape: {encoder_weights.shape}")
#32768 rows: latent features
# 512 columns: Each feature vector is in 512-dimensional space

# Step 1: Normalize each row (feature) to unit length
norms = encoder_weights.norm(dim=1, keepdim=True)  # Compute L2 norm of each row
normalized_weights = encoder_weights / norms  # Normalize rows to unit length

# Step 2: Compute the cosine similarity matrix
#Since we normalised, this is just dot product 
cosine_similarities = torch.matmul(normalized_weights, normalized_weights.T)

# Step 3: Print the result
print("Cosine Similarities:\n", cosine_similarities)
# np.savetxt("covariance_matrix.csv", cosine_similarities.numpy(), delimiter=",")

# Perform hierarchical clustering on the cosine similarity matrix
linkage_matrix = linkage(cosine_similarities[:5000, :5000], method='ward')

# Create a clustered heatmap using seaborn
sns.clustermap(cosine_similarities[:5000, :5000].numpy(), 
               cmap='coolwarm', 
               row_linkage=linkage_matrix, 
               col_linkage=linkage_matrix, 
               figsize=(10, 10))

plt.title("Clustered Cosine Similarity Heatmap (Sample)")
plt.show()