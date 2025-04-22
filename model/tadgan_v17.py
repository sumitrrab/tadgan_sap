import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

# -------------------------------
# 1) Basic setup
# -------------------------------
print("Starting TadGAN script...", flush=True)

# Device setup
print("Checking device...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Hyperparameters
epochs = 20
batch_size = 16
lr = 1e-4
seq_len = 15
print(f"Hyperparams -> epochs: {epochs}, batch_size: {batch_size}, lr: {lr}, seq_len: {seq_len}", flush=True)

# -------------------------------
# 2) Load data (first 50k rows)
# -------------------------------
data_path = "/home/stud447/timegan_processed_data.csv"
print(f"Reading CSV from {data_path} (first 50k rows)", flush=True)
df = pd.read_csv(data_path).iloc[:50000]
print(f"DataFrame loaded. Shape: {df.shape}", flush=True)

# Convert to float32
print("Converting data to float32...", flush=True)
data = df.values.astype(np.float32)
input_dim = data.shape[1]
print(f"Data has input_dim = {input_dim} features", flush=True)

# Sequence generation
print("Generating sequences...", flush=True)
sequences = np.array(
    [data[i:i + seq_len] for i in range(len(data) - seq_len + 1)], 
    dtype=np.float32
)
print(f"Sequences shape: {sequences.shape}", flush=True)

# -------------------------------
# 3) Split into train/test (Sequential Split)
# -------------------------------
print("Splitting into train/test (80/20 sequential)...", flush=True)
split_idx = int(len(sequences) * 0.8)
train_data = sequences[:split_idx]
test_data = sequences[split_idx:]  # Not used in training
print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}", flush=True)

# Convert to PyTorch tensors
train_tensor = torch.tensor(train_data).float()
print(f"train_tensor shape: {train_tensor.shape}", flush=True)

# Create DataLoader
print(f"Creating DataLoader with batch_size={batch_size}...", flush=True)
train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("DataLoader created!", flush=True)

# -------------------------------
# 4) Model definitions
# -------------------------------
print("Defining models (Generator & Discriminator)...", flush=True)

class Generator(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_dim, latent_dim, batch_first=True)
        self.decoder = torch.nn.LSTM(latent_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        latent, _ = self.encoder(x)
        reconstructed, _ = self.decoder(latent)
        return reconstructed

class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, latent_dim, batch_first=True)
        self.fc = torch.nn.Linear(latent_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

gen = Generator(input_dim).to(device)
disc = Discriminator(input_dim).to(device)
print("Models initialized on device.", flush=True)

# -------------------------------
# 5) Create optimizers & losses
# -------------------------------

print("Setting up optimizers and mixed-precision...", flush=True)
gen_opt = torch.optim.Adam(gen.parameters(), lr=5e-4)
disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-5)
criterion = torch.nn.BCEWithLogitsLoss()

# Mixed precision
scaler = GradScaler()

# -------------------------------
# 6) Warm-up test
# -------------------------------
print("Running warm-up check...", flush=True)

if len(train_loader) > 0:
    sample_batch = next(iter(train_loader))
    sample_input = sample_batch[0].to(device)
    with autocast():
        sample_fake = gen(sample_input)
        sample_out = disc(sample_fake)
    print("Warm-up check successful!", flush=True)
else:
    print("Warning: Train loader is empty! Skipping warm-up test.", flush=True)

# -------------------------------
# 7) Training Loop + Error Tracking
# -------------------------------
print("Starting training loop...", flush=True)

gen_losses = []
disc_losses = []
all_train_errors = []  # Renamed for clarity

for epoch in range(epochs):
    print(f"--- EPOCH {epoch + 1}/{epochs} ---", flush=True)

    epoch_disc_loss = 0.0
    epoch_gen_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        real = batch[0].to(device)

        # ---------------------------
        # 7a) Train Discriminator
        # ---------------------------
        disc_opt.zero_grad()

        with autocast():
            fake = gen(real)
            
            # Label smoothing
            real_labels = torch.ones(real.size(0), 1, dtype=torch.float32, device=device) * 0.85
            fake_labels = torch.zeros(real.size(0), 1, dtype=torch.float32, device=device) * 0.15
            
            disc_real = disc(real)
            loss_real = criterion(disc_real, real_labels)

            disc_fake = disc(fake.detach())
            loss_fake = criterion(disc_fake, fake_labels)

            disc_loss = (loss_real + loss_fake) * 0.5

        scaler.scale(disc_loss).backward()
        scaler.step(disc_opt)
        scaler.update()

        # ---------------------------
        # 7b) Train Generator
        # ---------------------------
        gen_opt.zero_grad()

        with autocast():
            gen_fake = disc(fake)
            
            # Generator target with label smoothing
            real_labels = torch.ones(real.size(0), 1, device=device) * 0.95
            mse_loss = torch.nn.MSELoss()(fake, real)
            gen_loss = criterion(gen_fake, real_labels) + 0.5 * mse_loss

        scaler.scale(gen_loss).backward()
        scaler.step(gen_opt)
        scaler.update()

        # ---------------------------
        # 7c) Track Losses
        # ---------------------------
        epoch_disc_loss += disc_loss.item()
        epoch_gen_loss += gen_loss.item()
        num_batches += 1

        # ---------------------------
        # 7d) Track reconstruction errors
        # ---------------------------
        with torch.no_grad():
            reconstruction_errors = torch.mean((fake - real) ** 2, dim=[1, 2])
            all_train_errors.extend(reconstruction_errors.cpu().numpy())

    # ---------------------------
    # 7e) Epoch reporting
    # ---------------------------
    avg_disc_loss = epoch_disc_loss / num_batches
    avg_gen_loss = epoch_gen_loss / num_batches

    gen_losses.append(avg_gen_loss)
    disc_losses.append(avg_disc_loss)

    print(f"End of Epoch {epoch + 1}/{epochs}: "
          f"Avg Disc Loss={avg_disc_loss:.4f}, "
          f"Avg Gen Loss={avg_gen_loss:.4f}", flush=True)

# -------------------------------
# 8) Post-training processing
# -------------------------------
# Calculate final threshold from ALL training errors
final_threshold = np.percentile(all_train_errors, 99)  # Changed to 99th percentile
print(f"\nFinal anomaly threshold (99th percentile): {final_threshold:.4f}", flush=True)

# Save models
torch.save(gen.state_dict(), "/home/stud447/tadgan_gen_v16.pth")
torch.save(disc.state_dict(), "/home/stud447/tadgan_disc_v16.pth")

# Save training results
pd.DataFrame({
    'gen_loss': gen_losses,
    'disc_loss': disc_losses
}).to_csv("/home/stud447/tadgan_training_results_v16.csv", index=False)

# Save ALL training errors for post-analysis
pd.DataFrame({'train_errors': all_train_errors}).to_csv(
    "/home/stud447/tadgan_train_errors_v16.csv", 
    index=False
)

print("Training complete! Models and training data saved.", flush=True)
