import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os
import tifffile
import h5py
from PIL import Image
import time
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score, f1_score

sample_name = 'HLCX022'

class PhikonEncoder(nn.Module):
    def __init__(self):
        super(PhikonEncoder, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        self.encoder = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, images):
        processed_images = []
        for img in images:  # Process each image in the batch
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()  # Convert tensor to numpy array
                img = np.transpose(img, (1, 2, 0))  # Change shape from (C, H, W) to (H, W, C)
                img = (img * 255).astype(np.uint8)  # Convert from [0, 1] range to [0, 255]
                img = Image.fromarray(img)  # Convert to PIL Image
            
            processed_images.append(img)

        # Process images using AutoImageProcessor
        inputs = self.image_processor(processed_images, return_tensors="pt").to(images.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs)

        return outputs.last_hidden_state[:, 0, :]  # Extract CLS token features (batch_size, 768)

# =============================
# Dataset for Histology Patches
# =============================
class HistologyDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, gene_expr_dir, transform=None):
        self.image_dir = image_dir
        self.gene_expr_dir = gene_expr_dir
        self.transform = transform
        self.samples = []
        self.expected_gene_dim = 0  # Will be determined from the first valid sample

        # Iterate over all .tif files in the image directory.
        for file in sorted(os.listdir(image_dir)):
            if file.endswith(".tif"):
                # Remove the "cell_" prefix and ".tif" extension to get the cell id.
                cell_id = file.replace("cell_", "").replace(".tif", "")
                gene_expr_file = os.path.join(gene_expr_dir, f"{cell_id}.h5")
                if os.path.exists(gene_expr_file):
                    self.samples.append((file, cell_id))
                else:
                    print(f"Gene expression file for {cell_id} not found in {gene_expr_dir}")
        
        # Determine the expected gene expression dimension using the first valid sample.
        if self.samples:
            sample_expr, gene_dim = self.load_gene_expression(os.path.join(gene_expr_dir, f"{self.samples[0][1]}.h5"))
            if gene_dim == 0:
                raise ValueError(f"No numeric gene expression data found in the first file: {self.samples[0][1]}.h5")
            self.expected_gene_dim = gene_dim

    def load_gene_expression(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            gene_data = {}
            # Iterate over keys and only include numeric datasets.
            for key in f.keys():
                data = f[key][:]
                if np.issubdtype(data.dtype, np.number):
                    gene_data[key] = data
            if len(gene_data) == 0:
                raise ValueError(f"No numeric gene expression data found in file: {h5_path}")
        
        # Create a DataFrame from the numeric gene data.
        df = pd.DataFrame(gene_data)
        df_numeric = df.select_dtypes(include=[np.number])
        expr_values = df_numeric.values.astype(np.float32)
        
        # If there's only one numeric column, convert it to a 1D vector.
        if expr_values.ndim == 2 and expr_values.shape[1] == 1:
            gene_dim = expr_values.shape[0]
            expr_values = expr_values.squeeze(axis=1)  # Now shape: (gene_dim,)
        else:
            gene_dim = expr_values.shape[1]

        # If expected_gene_dim is already set (i.e. not the first sample), adjust the dimension.
        if self.expected_gene_dim > 0:
            if gene_dim < self.expected_gene_dim:
                # For a 1D vector, pad at the end.
                if expr_values.ndim == 1:
                    expr_values = np.pad(expr_values, (0, self.expected_gene_dim - gene_dim), 'constant')
                else:
                    expr_values = np.pad(expr_values, ((0, 0), (0, self.expected_gene_dim - gene_dim)), 'constant')
            elif gene_dim > self.expected_gene_dim:
                # Truncate to expected dimension.
                if expr_values.ndim == 1:
                    expr_values = expr_values[:self.expected_gene_dim]
                else:
                    expr_values = expr_values[:, :self.expected_gene_dim]
            gene_dim = self.expected_gene_dim

        return expr_values, gene_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, cell_id = self.samples[idx]
        img_path = os.path.join(self.image_dir, image_file)
        expr_path = os.path.join(self.gene_expr_dir, f"{cell_id}.h5")
        
        # Load and process the image.
        img_array = tifffile.imread(img_path)
        if np.issubdtype(img_array.dtype, np.floating):
            img_array = (img_array * 255).astype(np.uint8)
        image = Image.fromarray(img_array).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load the gene expression data.
        gene_expr, _ = self.load_gene_expression(expr_path)
        # Ensure gene expression is a 1D vector
        if gene_expr.ndim != 1:
            gene_expr = gene_expr.squeeze()
        gene_expr = torch.tensor(gene_expr, dtype=torch.float)
        
        return image, gene_expr, cell_id

# =============================
# Data Preparation
# =============================
# Set these directories to your data locations.
image_dir = "/extra/zhanglab0/xil43/Xenium/CellNet_data/cell_patches/cell_img/" + sample_name
gene_expr_dir = "/extra/zhanglab0/xil43/Xenium/CellNet_data/cell_patches/cell_gene_expression/" + sample_name

# Define image transformations.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader.
dataset = HistologyDataset(image_dir, gene_expr_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

print(f"Total image patches in {image_dir}: {len(os.listdir(image_dir))}")
print(f"Valid patches with gene expression: {len(dataset)}")
print(f"Total batches in DataLoader: {len(dataloader)}")

# =============================
# Feature Extraction using ResNet50
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = PhikonEncoder().to(device)
model.eval()

features = []
patch_ids = []
gene_expressions = []

with torch.no_grad():
    for images, gene_exprs, patch_ids_batch in tqdm(dataloader, desc="Extracting Features"):
        images = images.to(device)
        outputs = model(images)
        features.append(outputs.cpu().numpy())
        patch_ids.extend(patch_ids_batch)
        gene_expressions.append(gene_exprs.numpy())

features = np.vstack(features)  # Shape: (num_valid_patches, 2048)
# Each gene expression is now a 1D array of length num_genes, so stacking yields shape: (num_valid_patches, num_genes)
gene_expressions = np.vstack(gene_expressions)
print("Feature extraction complete.")

# =============================
# PCA to Reduce Features to 256 Dimensions
# =============================
X_pca = PCA(n_components=256).fit_transform(features)
print('PCA Finished')

# =============================
# Train Ridge Regression and Compute Metrics
# =============================
num_genes = gene_expressions.shape[1]
results, ridge_train_times = {}, {}
for gene_idx in tqdm(range(num_genes), desc="Training Ridge Models"):
    start_time = time.time()
    Y, ridge = gene_expressions[:, gene_idx], Ridge(alpha=1.0)
    ridge.fit(X_pca, Y)
    Y_pred, pearson_corr = ridge.predict(X_pca), pearsonr(Y, ridge.predict(X_pca))[0]
    threshold, Y_true_bin, Y_pred_bin = np.median(Y), (Y > np.median(Y)).astype(int), (Y_pred > np.median(Y)).astype(int)
    accuracy, sensitivity = accuracy_score(Y_true_bin, Y_pred_bin), recall_score(Y_true_bin, Y_pred_bin)
    cm = confusion_matrix(Y_true_bin, Y_pred_bin)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        tn, fp, fn, tp = (cm[0, 0], 0, 0, 0) if Y_true_bin[0] == 0 else (0, 0, 0, cm[0, 0])
    else:
        raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    auc_score = roc_auc_score(Y_true_bin, Y_pred) if len(set(Y_true_bin)) > 1 else np.nan
    f1, elapsed_time = f1_score(Y_true_bin, Y_pred_bin), time.time() - start_time
    ridge_train_times[f"Gene_{gene_idx}"] = elapsed_time
    results[f"Gene_{gene_idx}"] = {"Pearson": pearson_corr, "Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity, "AUC": auc_score, "F1": f1}

# Compute overall average for each metric across all genes.
average_metrics = {metric: np.nanmean([results[f"Gene_{i}"][metric] for i in range(num_genes)]) for metric in ["Pearson", "Accuracy", "Sensitivity", "Specificity", "AUC", "F1"]}
results["Average"] = average_metrics

# Save Results
pd.DataFrame(results).T.to_csv("phikon_results.csv")
