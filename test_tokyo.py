import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

print("🗼 Loading Model for Tokyo Inference...")

# 1. Mac Device Setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 2. Load Model and Custom Weights
sam2_model = build_sam2("sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt", device=device)
checkpoint_path = "sam2_mac_checkpoint_epoch_4.pth"
sam2_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
sam2_model.eval()

# 3. Initialize the Automatic Mask Generator (Mac-Optimized)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,           # Back down to 32 (1,024 total points)
    points_per_batch=16,          # CRITICAL: Forces the Mac to process 16 points at a time instead of all at once!
    pred_iou_thresh=0.75,         # Keep the low confidence threshold
    stability_score_thresh=0.80,  
    min_mask_region_area=10       
)

# 4. Load your Tokyo Image
image_path = "tokyo.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("🔍 Scanning image and generating masks (this might take a minute...)")
with torch.no_grad():
    masks = mask_generator.generate(image_rgb)

print(f"✅ Found {len(masks)} potential structures!")

# 5. Visualization Setup
plt.figure(figsize=(12, 12))
plt.imshow(image_rgb)

# Overlay the generated masks
if len(masks) > 0:
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for mask in sorted_masks:
        m = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]]) # Random colors for each building
        img[m] = color_mask
    ax.imshow(img)

plt.axis('off')
plt.title("Fine-Tuned SAM 2: Tokyo Generalization Test", fontweight="bold")
plt.tight_layout()
plt.show()