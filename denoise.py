from torchvision.utils import save_image 
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm  # For process bar
import os

# ===================== Data preprocessing =====================
transform = transforms.Compose([
    transforms.ToTensor(),  # conversion to a tensor
    transforms.Normalize((0.5,), (0.5,))  # normalized to range [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Defining the diffusion process =====================
class Diffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)  # linear noise step
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)  

    def add_noise(self, x0, t):
        """add noise: q(x_t | x_0)"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noise = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, batch_size):
        """Random sampling timestep"""
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

# ===================== Defining the denoising network =====================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ===================== Train Model =====================
timesteps = 100
diffusion = Diffusion(timesteps=timesteps)
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

model_path = './model/model_weights.pth'# Check pre-trained model
if os.path.exists(model_path):
    print(f"Loading pretrained model from {model_path}")
    model.load_state_dict(torch.load(model_path))
else:
    print(f"No pretrained model found, starting training from scratch.")
    
epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, _ in progress_bar:
        images = images.to(device)

        # Random sampling timestep
        t = diffusion.sample_timesteps(images.size(0))

        # Add noise
        noisy_images, noise = diffusion.add_noise(images, t)

        # Predict Noise
        predicted_noise = model(noisy_images, t)

        # Calculate loss
        loss = criterion(predicted_noise, noise)
        epoch_loss += loss.item()

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Updating the suffix of the progress bar
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    print(f"Epoch Loss: {epoch_loss / len(train_loader):.4f}")
    
torch.save(model.state_dict(), model_path)
# ===================== Test denoise image =====================
def denoise_image(diffusion, model, noisy_image, timesteps):
    """stepwise denoising process：从 x_T -> x_0"""
    model.eval()
    with torch.no_grad():
        for t in tqdm(reversed(range(timesteps)), desc="Denoising", leave=False):
            # Parameters for the current timestep t
            beta_t = diffusion.betas[t]
            alpha_t = diffusion.alphas[t]
            alpha_hat_t = diffusion.alpha_hat[t]

            # Calculate x_{t-1}
            noise_pred = model(noisy_image, torch.tensor([t]).to(device))
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_hat_t)
            noisy_image = coef1 * (noisy_image - coef2 * noise_pred)

            # Adding a noise in non-final steps
            if t > 0:
                z = torch.randn_like(noisy_image)  # Random noise
                noisy_image += torch.sqrt(beta_t) * z

    return noisy_image


# Generate noisy images from dataset
test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_image, _ = next(iter(test_loader))
test_image = test_image.to(device)

# Add noise
t = torch.tensor([50]).to(device)  # set timestep
noisy_image, _ = diffusion.add_noise(test_image, t)

# Denoise
denoised_image = denoise_image(diffusion, model, noisy_image, timesteps)

# ===================== Results visualization =====================
save_image(test_image, "original_image.png", normalize=True)

# Save noise image
save_image(noisy_image, "noisy_image.png", normalize=True)

# Save Denoise image
save_image(denoised_image, "denoised_image.png", normalize=True)
