import os
import gc
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import kagglehub
from torch.utils.data import Dataset, DataLoader
from sam2.build_sam import build_sam2

print("🚀 Initiating Apple Silicon (MPS) SAM 2 Training...")

# ==========================================
# 1. Apple MPS Device Setup
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache() # Clear Apple Unified Memory
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ==========================================
# 2. Dataset & Loaders
# ==========================================
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

print("Locating Kaggle dataset...")
dataset_path = kagglehub.dataset_download("balraj98/massachusetts-buildings-dataset")
train_dataset = BatchedMassBuildingsDataset(os.path.join(dataset_path, "png", "train"), os.path.join(dataset_path, "png", "train_labels"))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

# ==========================================
# 3. Model Setup
# ==========================================
print("Loading Base SAM 2 Model...")
sam2_model = build_sam2("sam2_hiera_s.yaml", "/Users/arjunsingla/sam2/checkpoints/sam2_hiera_small.pt", device=device)
# 🧠 Load your custom trained weights to resume training!
previous_checkpoint = "sam2_mac_checkpoint_epoch_4.pth"
sam2_model.load_state_dict(torch.load(previous_checkpoint, map_location=device, weights_only=True))
print(f"Resuming training from {previous_checkpoint}...")
for name, param in sam2_model.named_parameters():
    if "sam_mask_decoder" in name or "sam_prompt_encoder" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam2_model.parameters()), lr=2e-5)
criterion_bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred).view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    return (1 - ((2. * intersection + smooth) / (union + smooth))).mean()

# ==========================================
# 4. Mac-Safe Training Loop
# ==========================================
epochs = 4
CHUNK_SIZE = 5         # Keep this small. Mac Unified Memory shares RAM with Chrome!
MAX_BOXES_PER_IMG = 30 # Sub-sampling for speed

print("\nStarting Training...")
for epoch in range(epochs):
    sam2_model.train()
    epoch_loss = 0.0
    
    for step, (images, masks, multi_boxes) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device).float() 
        optimizer.zero_grad()
        
        with torch.no_grad():
            backbone_out = sam2_model.forward_image(images)
            _, vision_feats, _, _ = sam2_model._prepare_backbone_features(backbone_out)
            feats = [feat.permute(1, 2, 0).view(images.size(0), -1, *s) for feat, s in zip(vision_feats, [(256, 256), (128, 128), (64, 64)])]
            image_embed = feats[-1]
            high_res_feats = feats[:-1] if sam2_model.use_high_res_features_in_sam else None

        for i in range(images.size(0)):
            single_image_embed = image_embed[i].unsqueeze(0)
            single_high_res_feats = [f[i].unsqueeze(0) for f in high_res_feats] if high_res_feats else None
            single_gt_mask = masks[i] 
            
            img_boxes = multi_boxes[i].to(device)
            num_boxes = img_boxes.size(0)
            if num_boxes == 0 or (num_boxes == 1 and torch.all(img_boxes == 0)): continue
            
            if num_boxes > MAX_BOXES_PER_IMG:
                rand_indices = torch.randperm(num_boxes, device=device)[:MAX_BOXES_PER_IMG]
                img_boxes = img_boxes[rand_indices]
                num_boxes = MAX_BOXES_PER_IMG
                
            num_chunks = (num_boxes + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            for chunk_start in range(0, num_boxes, CHUNK_SIZE):
                chunk_boxes = img_boxes[chunk_start : chunk_start + CHUNK_SIZE]
                num_chunk_boxes = chunk_boxes.size(0)
                
                batched_embed = single_image_embed.repeat(num_chunk_boxes, 1, 1, 1)
                batched_high_res = [f.repeat(num_chunk_boxes, 1, 1, 1) for f in single_high_res_feats] if single_high_res_feats else None
                
                sparse, dense = sam2_model.sam_prompt_encoder(points=None, boxes=chunk_boxes, masks=None)
                
                low_res_masks, _, _, _ = sam2_model.sam_mask_decoder(
                    image_embeddings=batched_embed,
                    image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=batched_high_res
                )
                
                pred_masks_up = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False).squeeze(1)
                chunk_targets = torch.zeros_like(pred_masks_up)
                
                for k in range(num_chunk_boxes):
                    x1, y1, x2, y2 = chunk_boxes[k].int().tolist()
                    x1, x2 = max(0, x1), min(1024, x2)
                    y1, y2 = max(0, y1), min(1024, y2)
                    chunk_targets[k, y1:y2, x1:x2] = single_gt_mask[y1:y2, x1:x2]
                
                loss = criterion_bce(pred_masks_up, chunk_targets) + dice_loss(pred_masks_up, chunk_targets)
                scaled_loss = loss / (num_chunks * images.size(0))
                
                scaled_loss.backward()
                epoch_loss += scaled_loss.item()
            
        optimizer.step()
            
        if step % 10 == 0:
            current_avg_loss = epoch_loss / (step + 1)
            print(f"Epoch {epoch+1} | Step {step} | Average Batch Loss: {current_avg_loss:.4f}")
            
    print(f"=== Epoch {epoch+1} Complete ===")
    
    # 💾 THE LIFESAVER: Save progress locally
    save_path = f"sam2_mac_PHASE2_epoch_{epoch+1}.pth"
    torch.save(sam2_model.state_dict(), save_path)
    print(f"💾 Model safely saved to: {save_path}")