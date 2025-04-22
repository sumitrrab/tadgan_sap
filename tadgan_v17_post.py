import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# -------------------------------
# 1) Basic setup
# -------------------------------
print("Starting TadGAN post-analysis...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2) Load trained generator
# -------------------------------
class Generator(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_dim, latent_dim, batch_first=True)
        self.decoder = torch.nn.LSTM(latent_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        latent, _ = self.encoder(x)
        return self.decoder(latent)[0]

print("Loading trained generator...", flush=True)
input_dim = 285  # Must match training config
gen = Generator(input_dim).to(device)
gen.load_state_dict(torch.load("/home/stud447/tadgan_gen_v17.pth"))
gen.eval()

# -------------------------------
# 3) Load training threshold
# -------------------------------
print("Loading training threshold...", flush=True)
train_errors = pd.read_csv("/home/stud447/tadgan_train_errors_v17.csv")['train_errors'].values
threshold = np.percentile(train_errors, 99)  # Using top 1% anomaly threshold
print(f"Using threshold from training data (99th percentile): {threshold:.4f}", flush=True)

# -------------------------------
# 4) Prepare TEST data
# -------------------------------
print("Preparing test data...", flush=True)
data_path = "/home/stud447/timegan_processed_data.csv"

# Load full dataset and process EXACTLY like training
full_data = pd.read_csv(data_path).iloc[:50000].values.astype(np.float32)
sequences = np.array(
    [full_data[i:i+15] for i in range(len(full_data)-15+1)],
    dtype=np.float32
)

# Take last 20% as test (matches training split)
test_data = sequences[int(len(sequences)*0.8):]
test_tensor = torch.tensor(test_data).float().to(device)
test_loader = DataLoader(TensorDataset(test_tensor), batch_size=16, shuffle=False)
print(f"Test data shape: {test_data.shape}", flush=True)

# -------------------------------
# 5) Compute TEST reconstruction errors
# -------------------------------
print("Computing test reconstruction errors...", flush=True)
test_errors = []

with torch.no_grad():
    for batch in test_loader:
        real = batch[0].to(device)
        fake = gen(real)
        error = torch.mean((fake - real)**2, dim=[1,2])
        test_errors.extend(error.cpu().numpy())

# -------------------------------
# 6) Save results
# -------------------------------
results = pd.DataFrame({
    'test_reconstruction_error': test_errors,
    'threshold': threshold  # From training data
})
results.to_csv("/home/stud447/tadgan_post_results_v17.csv", index=False)
print("Results saved with training-derived threshold", flush=True)

# -------------------------------
# 7) Visualization (No metrics without labels)
# -------------------------------
# Histogram
plt.figure(figsize=(10, 6))
plt.hist(test_errors, bins=50, alpha=0.7, color='blue', label='Test Errors')
plt.axvline(threshold, color='red', linestyle='--', label='Training Threshold')
plt.title("Test Reconstruction Errors vs Training Threshold")
plt.xlabel("Reconstruction Error")
plt.legend()
plt.show()

# Time Series
plt.figure(figsize=(12, 6))
plt.plot(test_errors, label='Test Reconstruction Error')
plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Error Over Time (Test Data)")
plt.ylabel("Error")
plt.xlabel("Time Step")
plt.legend()
plt.show()

print("Post-analysis complete! Focus on points above threshold.", flush=True)
