import torch
import numpy as np
from make_agent_id_embedding_model import LinearVAE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

N_AGENT = 11
OBS_DIM = 217
ACTION_DIM = 19
LATENT_DIM = 4
EXTRA_DECODER_INPUT = OBS_DIM + ACTION_DIM
RECONSTRUCT_SIZE = OBS_DIM + 1
ENCODER_INPUT_SIZE = N_AGENT

help_encoder_model = LinearVAE(
    features=LATENT_DIM,
    input_size=ENCODER_INPUT_SIZE,
    extra_decoder_input=EXTRA_DECODER_INPUT,
    reconstruct_size=RECONSTRUCT_SIZE
)

transformer_state_dict = torch.load(
    "/MountPoint/enc_help_transformer/40.pt",
    map_location="cpu"
)
help_encoder_model.load_state_dict(transformer_state_dict)
help_encoder_model.eval()

eye = np.eye(N_AGENT, dtype=np.float32) 

with torch.no_grad():
    encoded_mu, encoded_log_var = help_encoder_model.encode(eye)

print("Encoded mu shape:", encoded_mu.shape)
print("Encoded log_var shape:", encoded_log_var.shape)

emb = encoded_mu.detach().cpu().numpy()
print("Embedding shape (numpy):", emb.shape)

colors = plt.cm.tab10(np.linspace(0, 1, N_AGENT))

pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb)

print("Explained variance ratio (2D PCA):", pca.explained_variance_ratio_)

plt.figure(figsize=(6, 5))
for i in range(emb_2d.shape[0]):
    x, y = emb_2d[i]
    plt.scatter(x, y, color=colors[i], label=f"Agent {i}")
    plt.text(x + 0.01, y + 0.01, str(i), fontsize=9)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Agent ID Encodings (2D)")
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)

handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), loc='upper left')

plt.tight_layout()
plt.savefig("2d-visualize-agent-id-embed.png", dpi=200)
plt.close()

# #######################################
# # ----------- 3D PCA Plot ------------
# #######################################

# # With LATENT_DIM = 3, PCA to 3D is basically a rotation; still fine for visualisation.
# pca_3d = PCA(n_components=3)
# emb_3d = pca_3d.fit_transform(emb)

# fig = plt.figure(figsize=(7, 6))
# ax = fig.add_subplot(111, projection='3d')

# for i in range(emb_3d.shape[0]):
#     x, y, z = emb_3d[i]
#     ax.scatter(x, y, z, color=colors[i], label=f"Agent {i}")
#     ax.text(x + 0.01, y + 0.01, z + 0.01, str(i), fontsize=8)

# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
# ax.set_title("3D PCA of Agent ID Encodings")

# # Legend (remove duplicates)
# handles, labels = ax.get_legend_handles_labels()
# unique = dict(zip(labels, handles))
# ax.legend(unique.values(), unique.keys(), loc='upper left')

# plt.tight_layout()
# plt.savefig("3d-visualize-agent-id-embed.png", dpi=200)
# plt.close()

threshold = 0.2

for k in range(2, min(7, N_AGENT + 1)): 
    print(f"\n===== GMM with K = {k} components =====")

    gmm = GaussianMixture(
        n_components=k,
        random_state=0,
        n_init=10,
        covariance_type="full"
    )
    gmm.fit(emb)

    responsibilities = gmm.predict_proba(emb)

    hard_labels = responsibilities.argmax(axis=1)

    for agent_id in range(N_AGENT):
        probs = responsibilities[agent_id]
        overlapping_clusters = [c for c, p in enumerate(probs) if p >= threshold]

        probs_str = ", ".join(
            [f"C{c}: {p:.2f}" for c, p in enumerate(probs)]
        )
        overlap_str = ", ".join([f"C{c}" for c in overlapping_clusters]) or "None"

        print(
            f"Agent {agent_id}: "
            f"hard_cluster={hard_labels[agent_id]}, "
            f"overlapping_clusters=[{overlap_str}], "
            f"probs=[{probs_str}]"
        )
    try:
        if k < emb.shape[0]:
            score = silhouette_score(emb, hard_labels)
            print(f"Silhouette score (using hard labels): {score:.4f}")
        else:
            print("Silhouette score cannot be computed (k >= number of samples).")
    except:
        pass