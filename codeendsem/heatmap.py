import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("\n--- Generating Groundwater Inference Heatmap (RF Hybrid) ---")

# 1. Simulate a 10x10 geographical grid (100 patches) from the test set
grid_size = 10
num_patches = grid_size * grid_size

# Grab the first 100 patches
sample_patches = X_p_test[:num_patches]
sample_tabular = X_t_test[:num_patches]

# 2. Predict probabilities directly using your model object
print("Calculating groundwater probabilities...")
# Because of how you wrote predict_proba, this automatically extracts 
# the deep features, fuses them, and returns the probabilities!
suitability_probs = model.predict_proba(sample_patches, sample_tabular)

# 3. Reshape the 1D list of 100 probabilities into a 2D 10x10 grid
heatmap_grid = suitability_probs.reshape(grid_size, grid_size)

# 4. Plot the Deployment Heatmap
plt.figure(figsize=(10, 8))
# YlGnBu colormap: Yellow = Dry/Not Suitable, Blue = Water/Suitable
sns.heatmap(heatmap_grid, cmap='YlGnBu', vmin=0.0, vmax=1.0, 
            cbar_kws={'label': 'Probability of Groundwater Suitability'})

plt.title('Predictive Heatmap: Groundwater Potential (ResNet + RF)', fontsize=14, pad=15)
plt.xlabel('Grid X (2km Spatial Resolution)')
plt.ylabel('Grid Y (2km Spatial Resolution)')

plt.tight_layout()
plt.show()

print("Success! Right-click and save this image for the 'Deployment' section of your report.")