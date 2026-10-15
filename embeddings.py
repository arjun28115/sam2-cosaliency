import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Force local repository root path visibility to ensure safe, stable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Model Registries & Local Imports ---
from segment_anything import sam_model_registry, SamPredictor       # SAM 1
from sam2.build_sam import build_sam2                               # SAM 2
from sam2.sam2_image_predictor import SAM2ImagePredictor             # SAM 2
from ultralytics.models.sam import SAM3SemanticPredictor            # SAM 3

# =====================================================================
# 1. Unified Feature Extractor Optimized for Mac Environment
# =====================================================================
class UnifiedSAMFeatureExtractor:
    def __init__(self, device="mps"):
        self.device = device
        
        # Configure weight paths checking local directory root
        sam1_path = "sam_vit_h.pth"
        sam2_path = "sam2_hiera_large.pt"
        sam3_path = "sam3.pt"
        
        # Fallback to CPU if weights are missing to prevent silent runtime failure
        for p in [sam1_path, sam2_path, sam3_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required local checkpoint file: '{p}'. Please place it in this folder.")

        print(f"----> Initializing SAM 1 (ViT-Heavy) on {device}...")
        self.sam1 = sam_model_registry["vit_h"](checkpoint=sam1_path).to(device)
        self.predictor_s1 = SamPredictor(self.sam1)
        
        print(f"----> Initializing SAM 2 (Hiera-Large) using local configs...")
        # Point directly to your local repository configuration architecture
        self.sam2 = build_sam2("sam2_hiera_l.yaml", sam2_path, device=device)
        self.predictor_s2 = SAM2ImagePredictor(self.sam2)
        
        print(f"----> Initializing SAM 3 (Perception Vision-Language)...")
        self.predictor_s3 = SAM3SemanticPredictor(overrides=dict(model=sam3_path, device=device))

    def get_sam1_features(self, image_np):
        self.predictor_s1.set_image(image_np)
        with torch.no_grad():
            feats = self.predictor_s1.features.squeeze(0).permute(1, 2, 0)
        return feats  # (64, 64, 256)

    def get_sam2_features(self, image_np):
        self.predictor_s2.set_image(image_np)
        with torch.no_grad():
            feats = self.predictor_s2._features["high_res_feats"][0].squeeze(0).permute(1, 2, 0)
        return feats  # (256, 256, C)

    def get_sam3_features(self, image_np):
        self.predictor_s3.set_image(image_np)
        with torch.no_grad():
            if isinstance(self.predictor_s3.features, dict):
                if "high_res_feats" in self.predictor_s3.features:
                    s3_raw = self.predictor_s3.features["high_res_feats"][0]
                elif "image_embed" in self.predictor_s3.features:
                    s3_raw = self.predictor_s3.features["image_embed"]
                else:
                    s3_raw = next(iter(self.predictor_s3.features.values()))
                    if isinstance(s3_raw, list): s3_raw = s3_raw[0]
            else:
                s3_raw = self.predictor_s3.features
            feats = s3_raw.squeeze(0).permute(1, 2, 0)
        return feats  # Dynamic resolution layout handling padding aspect ratio safely

# =====================================================================
# 2. Main Executive Latent Dimensionality Map Pipeline
# =====================================================================
def run_local_mac_pipeline(image_filename="HD_home.jpg", n_clusters=5):
    # Auto-detect Apple Silicon acceleration hardware (MPS), fallback to standard CPU if unavailable
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("Warning: Apple Silicon MPS hardware accelerator not detected. Falling back to slow CPU mode.")
        device = torch.device("cpu")
        
    if not os.path.exists(image_filename):
        raise FileNotFoundError(f"Target file '{image_filename}' not found. Place it inside the current sam2 directory root.")

    img = Image.open(image_filename).convert("RGB")
    img_np = np.array(img)
    
    # Run structural wrapper sequence
    extractor = UnifiedSAMFeatureExtractor(device=device)
    
    print("\nExtracting cross-generational latent layers...")
    features_dict = {
        'SAM 1': extractor.get_sam1_features(img_np),
        'SAM 2': extractor.get_sam2_features(img_np),
        'SAM 3': extractor.get_sam3_features(img_np)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    models = ['SAM 1', 'SAM 2', 'SAM 3']
    
    for idx, name in enumerate(models):
        feats_tensor = features_dict[name]
        h_f, w_f, c_f = feats_tensor.shape
        
        # Flatten feature matrices down to tokens
        flat_features = feats_tensor.reshape(-1, c_f).cpu().numpy()
        
        # Vector standardization normalization 
        flat_features = flat_features / (np.linalg.norm(flat_features, axis=-1, keepdims=True) + 1e-8)
        
        # PCA Dimensionality Reduction
        print(f"[{name}] Reducing features via PCA to 2D...")
        pca = PCA(n_components=2, random_state=42)
        projected_2d = pca.fit_transform(flat_features)
        
        # K-Means Token Partitioning
        print(f"[{name}] Isolating K-Means Cluster Boundaries...")
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(flat_features)
        
        # Row 1: Latent Vector Fields Scattering
        ax_scatter = axes[0, idx]
        stride = max(1, len(projected_2d) // 4000) # Stride throttling to keep Mac plotting memory low
        
        ax_scatter.scatter(
            projected_2d[::stride, 0], 
            projected_2d[::stride, 1], 
            c=cluster_labels[::stride], 
            cmap='jet', 
            alpha=0.5, 
            s=5
        )
        ax_scatter.set_title(f"{name} Latent Topography\n(Local PCA View)", weight='bold', fontsize=12)
        ax_scatter.grid(True, linestyle='--', alpha=0.3)
        
        # Row 2: Image Re-mapping Mask Visualizer
        ax_mask = axes[1, idx]
        spatial_cluster_mask = cluster_labels.reshape(h_f, w_f)
        
        ax_mask.imshow(spatial_cluster_mask, cmap='jet')
        ax_mask.set_title(f"{name} Unsupervised Segments\n({w_f}x{h_f} Feature Grid)", weight='bold', fontsize=12)
        ax_mask.axis('off')
        
    plt.suptitle(f"Mac Local Latent Projection Engine: HD_home.jpg", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    
    # Save a high-res backup copy to your local root directory
    output_filename = "sam_latent_benchmark_results.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nSuccess! Summary visualization chart saved locally to your folder as: '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    run_local_mac_pipeline("HD_home.jpg", n_clusters=5)