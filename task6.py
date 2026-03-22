import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("3) Sentiment dataset.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Use only numerical columns for clustering
num_cols = ['Retweets', 'Likes', 'Hour', 'Month', 'Year', 'Day']
df_cluster = df[num_cols].copy()

# Fill missing values
df_cluster.fillna(df_cluster.mean(), inplace=True)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# --- Elbow Method to find best K ---
print("\nFinding best K using Elbow Method...")
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', color='blue')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Finding Optimal K')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('task6_elbow.png')
plt.show()
print("Elbow plot saved as task6_elbow.png")

# --- Apply K-Means with best K=3 ---
print("\n--- Applying K-Means with K=3 ---")
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)

print("\nCluster counts:")
print(df['Cluster'].value_counts().sort_index())

print("\nCluster averages:")
print(df.groupby('Cluster')[num_cols].mean().round(2))

# --- Visualize clusters using PCA (reduce to 2D) ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for i in range(3):
    mask = df['Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], label=f'Cluster {i}',
                alpha=0.6, s=50)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters - Sentiment Data')
plt.legend()
plt.tight_layout()
plt.savefig('task6_clusters.png')
plt.show()
print("Cluster plot saved as task6_clusters.png")
print("\nTask 6 Complete!")