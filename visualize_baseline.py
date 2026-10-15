import os
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
from torch.utils.data import Dataset, DataLoader
from sam2.build_sam import build_sam2

print("🔍 Loading Fine-Tuned Model for Visualization...")

# 1. Mac Device Setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 2. Rebuild Dataset Logic (Validation Set Only)
class BatchedMassBuildingsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=1024, min_area=30):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.min_area = min_area
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.tif'))])

    def __len__(self): return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_name)), cv2.COLOR_BGR2RGB), (self.image_size, self.image_size))
        mask = cv2.resize(cv2.imread(os.path.join(self.mask_dir, img_name), cv2.IMREAD_GRAYSCALE), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([float(x), float(y), float(x+w), float(y+h)])
        if not boxes: boxes = [[0.0, 0.0, float(self.image_size), float(self.image_size)]]
            
        return (torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, 
                torch.from_numpy(binary_mask).long(), 
                torch.tensor(boxes, dtype=torch.float32))

def custom_collate(batch):
    return (torch.stack([item[0] for item in batch]), 
            torch.stack([item[1] for item in batch]), 
            [item[2] for item in batch])

dataset_path = kagglehub.dataset_download("balraj98/massachusetts-buildings-dataset")
val_dataset = BatchedMassBuildingsDataset(os.path.join(dataset_path, "png", "val"), os.path.join(dataset_path, "png", "val_labels"))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

# 3. Load Base Model AND Your Custom Weights
sam2_model = build_sam2("sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt", device=device)

# Skip loading the custom brain to see the raw baseline model!
# checkpoint_path = "sam2_mac_checkpoint_epoch_4.pth"
# sam2_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
sam2_model.eval() # Lock the model into inference mode

# 4. Visualization Logic
images, masks, multi_boxes = next(iter(val_loader)) 
num_samples = 2 
fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
CHUNK_SIZE = 15 # Can be larger for inference since gradients aren't stored

print("🎨 Rendering Predictions...")
with torch.no_grad():
    for i in range(num_samples):
        single_img = images[i:i+1].to(device)
        gt_mask = masks[i].numpy()
        img_boxes = multi_boxes[i].to(device)
        num_boxes = img_boxes.size(0)
        
        backbone_out = sam2_model.forward_image(single_img)
        _, vision_feats, _, _ = sam2_model._prepare_backbone_features(backbone_out)
        feats = [feat.permute(1, 2, 0).view(1, -1, *s) for feat, s in zip(vision_feats, [(256, 256), (128, 128), (64, 64)])]
        
        all_chunked_masks = []
        for chunk_start in range(0, num_boxes, CHUNK_SIZE):
            chunk_boxes = img_boxes[chunk_start : chunk_start + CHUNK_SIZE]
            num_chunk_boxes = chunk_boxes.size(0)
            
            batched_embed = feats[-1].repeat(num_chunk_boxes, 1, 1, 1)
            batched_high_res = [f.repeat(num_chunk_boxes, 1, 1, 1) for f in feats[:-1]] if sam2_model.use_high_res_features_in_sam else None
            
            sparse, dense = sam2_model.sam_prompt_encoder(points=None, boxes=chunk_boxes, masks=None)
            low_res_masks, _, _, _ = sam2_model.sam_mask_decoder(
                image_embeddings=batched_embed, image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False, repeat_image=False, high_res_features=batched_high_res
            )
            all_chunked_masks.append(low_res_masks)
            
        full_low_res_masks = torch.cat(all_chunked_masks, dim=0)
        pred_masks_up = F.interpolate(full_low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False).squeeze(1)
        
        combined_pred, _ = torch.max(pred_masks_up, dim=0)
        pred_mask = (combined_pred.cpu().numpy() > 0.0).astype(np.uint8) # Threshold at 0.0 for logits
        
        img_np = single_img.squeeze().cpu().permute(1, 2, 0).numpy()
        def apply_overlay(ax, mask, color_rgb):
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask == 1] = color_rgb + (0.5,)
            ax.imshow(overlay)
        
        axes[i, 0].imshow(img_np); axes[i, 0].set_title(f"Raw Image ({num_boxes} Buildings)", fontweight="bold"); axes[i, 0].axis('off')
        axes[i, 1].imshow(img_np); apply_overlay(axes[i, 1], gt_mask, (0.1, 0.9, 0.1)); axes[i, 1].set_title("Ground Truth", fontweight="bold"); axes[i, 1].axis('off')
        axes[i, 2].imshow(img_np); apply_overlay(axes[i, 2], pred_mask, (0.1, 0.4, 0.9)); axes[i, 2].set_title("Without Fine-Tuning", fontweight="bold"); axes[i, 2].axis('off')

plt.tight_layout()
plt.show()