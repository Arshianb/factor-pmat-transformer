import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances, silhouette_score

file_paths = sorted(glob.glob("/MountPoint/help_enc_output/*"))
mean_obs = []

for path in file_paths:
    npy_file = np.load(path)   
    mean_obs.append(npy_file)
out = np.mean(np.mean(np.array(mean_obs), 0), 0)
import numpy as np
kmeans = KMeans(n_clusters=2, n_init=100, random_state=0)
kmeans.fit(out)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Cluster assignments:", labels)

intra_distances = {}
for cluster_id in [0, 1]:
    idx = np.where(labels == cluster_id)[0]
    pts = out[idx]

    if len(idx) == 1:
        intra_distances[cluster_id] = 0.0
    else:
        dists = pairwise_distances(pts)
        intra_distances[cluster_id] = dists[np.triu_indices(len(pts), 1)].mean()

inter_distance = np.linalg.norm(centroids[0] - centroids[1])
print("Intra-cluster distances:")
for cid, dist in intra_distances.items():
    print(f"  Cluster {cid}: {dist}")
print("\nInter-cluster distance (centroids):", inter_distance)
sil = silhouette_score(out, labels) 
sil_percent = (sil + 1) / 2 * 100 

print(f"\nSilhouette score: {sil:.4f}")
print(f"Silhouette % (0–100): {sil_percent:.2f}%")
avg_intra = np.mean(list(intra_distances.values()))
if avg_intra == 0:
    separation_ratio = np.inf
else:
    separation_ratio = inter_distance / avg_intra
print(f"\nSeparation ratio (inter / avg_intra): {separation_ratio:.4f}")


file_paths = sorted(glob.glob("/MountPoint/help_enc_output/*"))
mean_obs = []

for path in file_paths:
    npy_file = np.load(path)
    mean_obs.append(npy_file)

mean_obs = np.stack(mean_obs, axis=0)

file_embeddings = mean_obs.mean(axis=(1, 2))

pca = PCA(n_components=2)
X_2d = pca.fit_transform(file_embeddings) 

print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(6, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1])

for i, path in enumerate(file_paths):
    plt.annotate(str(i), (X_2d[i, 0], X_2d[i, 1]))

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of mean_obs (one point per npy file)")
plt.tight_layout()
plt.savefig("visualization.png")