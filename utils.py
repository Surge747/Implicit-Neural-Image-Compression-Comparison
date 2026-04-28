import os
import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# --- 1. SETUP FOLDERS AUTOMATICALLY ---
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- 2. THE METRICS ENGINE ---
class MetricsEngine:
    def __init__(self, device='cpu'):
        self.device = device
        # Try to load LPIPS, but don't crash if it's not installed
        try:
            import lpips
            self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
            self.has_lpips = True
            print("LPIPS loaded successfully.")
        except ImportError:
            self.has_lpips = False
            print("LPIPS not found. Run 'pip install lpips' if you want it.")

    def compute_all(self, img_true, img_pred):
        """ Expects (H, W, 3) RGB numpy arrays in [0, 255] """
        
        # 1. PSNR (Math metric)
        mse = np.mean((img_true.astype(np.float64) - img_pred.astype(np.float64)) ** 2)
        psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        
        # 2. SSIM (Structural metric)
        ssim_val = ssim(img_true, img_pred, channel_axis=-1, data_range=255)
        
        # 3. NCE (Normalized Cross Correlation)
        img_true_flat = img_true.flatten()
        img_pred_flat = img_pred.flatten()
        nce = np.corrcoef(img_true_flat, img_pred_flat)[0, 1]

        # 4. LPIPS (Perceptual Deep Learning metric)
        lpips_val = 0.0
        if self.has_lpips:
            # Convert to tensors in range [-1, 1] for LPIPS
            t_true = torch.tensor(img_true).permute(2,0,1).unsqueeze(0).to(self.device) / 127.5 - 1
            t_pred = torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device) / 127.5 - 1
            with torch.no_grad():
                lpips_val = self.loss_fn_vgg(t_true, t_pred).item()

        return {"PSNR": psnr, "SSIM": ssim_val, "NCE": nce, "LPIPS": lpips_val}

# --- 3. THE VISUAL COMPARISON TOOL ---
def visual_compare(img_path_a, img_path_b, title_a="Original", title_b="Compressed"):
    """ Use this to compare any two images and get a heatmap. """
    img_a = cv2.cvtColor(cv2.imread(img_path_a), cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(cv2.imread(img_path_b), cv2.COLOR_BGR2RGB)
    
    engine = MetricsEngine()
    metrics = engine.compute_all(img_a, img_b)
    
    error_map = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32)).mean(axis=-1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_a)
    axes[0].set_title(title_a)
    axes[0].axis("off")
    
    metrics_text = f"PSNR: {metrics['PSNR']:.2f} | SSIM: {metrics['SSIM']:.3f}\nNCE: {metrics['NCE']:.3f} | LPIPS: {metrics['LPIPS']:.3f}"
    axes[1].imshow(img_b)
    axes[1].set_title(f"{title_b}\n{metrics_text}")
    axes[1].axis("off")
    
    im3 = axes[2].imshow(error_map, cmap='inferno')
    axes[2].set_title("Error Heatmap (Brighter = Worse)")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()