import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
beta_start = 1e-4
beta_end = 0.02
t = 1000 ##timesteps
betas = torch.linspace(beta_start, beta_end, t, device=device)
## How much of original image to keep at each timestep
alphas = 1. - betas
## Cumulative product of all alpha values for beta linearly ditributed from start to end
alphas_cumprod = torch.cumprod(alphas, dim=0)
## sqrt of cumulative product of alpha (root of alpha t , root of (1-alpha t))
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def forward_diffusion_sample(x_0, t, device):
    ## value of epsilon - normal distribution with mean =0, variance =1
    #Returns a tensor with the same size as input that is filled with random numbers 
    # #from a normal distribution with mean 0 and variance 1
    noise = torch.randn_like(x_0)
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise

##load images using datasets , batching using dataloader

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')  # Load as RGB
        
        if self.transform:
            image = self.transform(image)
            
        return image

# Parameters
img_dir = 'F:/ai/Langchain/img_align_celeba/img_align_celeba'  # Replace with your image folder
batch_size = 64
image_size = 64  # Target image size
T = 1000  # Total timesteps
beta_start=1e-4
beta_end=0.02

# Define transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),  # Converts to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scales to [-1, 1]
])

# Create dataset and dataloader
dataset = ImageDataset(img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up diffusion parameters (same as before)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


for batch_idx, x_0 in enumerate(dataloader):
    x_0 = x_0.to(device)
    
    # Sample random timesteps for each image in batch
    t = torch.randint(0, T, (x_0.shape[0],), device=device).long()
    
    # Apply forward diffusion
    x_t, noise = forward_diffusion_sample(
        x_0=x_0,
        t=t,
        device=device,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod
    )

# Now you can:
    # 1. Pass x_t to your model
    # 2. Train the model to predict 'noise'
    # 3. Compute loss between predicted and actual noise
    
    print(f"Batch {batch_idx}:")
    print(f"Original shape: {x_0.shape} | Noisy shape: {x_t.shape}")
    print(f'original image tensor: {x_0} | Noisy image : {x_t}')
    print(f"Timesteps range: {t.min().item()} to {t.max().item()}")
    
    # For demonstration, break after first batch
    if batch_idx == 10:
        break


import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

# Assuming we have the previous forward diffusion code and UNet definition
# Let's add the reverse diffusion sampling function

@torch.no_grad()
def reverse_diffusion_sample(model, image_size, batch_size=8, channels=3, T=1000, 
                           device='cuda', betas=None, alphas_cumprod=None):
    """
    Generates samples from random noise using the reverse diffusion process.
    
    Args:
        model: Trained denoising U-Net model
        image_size: Size of the output images
        batch_size: Number of images to generate
        channels: Number of channels in images
        T: Total timesteps
        device: Device to run on
        betas: Noise schedule (from forward process)
        alphas_cumprod: Cumulative product of alphas (from forward process)
    """
    # Prepare model and initial noise
    model.eval()
    img = torch.randn((batch_size, channels, image_size, image_size), device=device)
    
    # Calculate required coefficients if not provided
    if betas is None or alphas_cumprod is None:
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # Reverse process loop
    for t in reversed(range(T)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise using the model
        pred_noise = model(img, t_batch)
        
        # Calculate coefficients for this timestep
        alpha_t = alphas[t]
        alpha_t_cumprod = alphas_cumprod[t]
        beta_t = betas[t]
        
        # Compute x_{t-1}
        img = sqrt_recip_alphas[t] * (img - ((1 - alpha_t) / sqrt_one_minus_alphas_cumprod[t]) * pred_noise)
        
        # Add noise except at last step
        if t > 0:
            noise = torch.randn_like(img)
            img += torch.sqrt(beta_t) * noise
    
    # Clip to [-1, 1] and rescale to [0, 1] for saving
    img = torch.clamp(img, -1., 1.)
    img = (img + 1) / 2  # Scale from [-1,1] to [0,1]
    return img

# Example usage with previous code
if __name__ == "__main__":
    # 1. Initialize model (assuming UNet is defined as before)
    model = UNet().to(device)
    
    # 2. Load trained weights (if available)
    # model.load_state_dict(torch.load('diffusion_model.pth'))
    
    # 3. Generate samples
    num_samples = 8
    generated_images = reverse_diffusion_sample(
        model=model,
        image_size=image_size,
        batch_size=num_samples,
        channels=channels,
        T=T,
        device=device,
        betas=betas,
        alphas_cumprod=alphas_cumprod
    )
    
    # 4. Save results
    os.makedirs('samples', exist_ok=True)
    save_image(generated_images, 'samples/generated.png', nrow=4)
    print("Generated samples saved to 'samples/generated.png'")
