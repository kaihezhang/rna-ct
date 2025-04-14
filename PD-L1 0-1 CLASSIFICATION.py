import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn as nn
import nibabel as nib
import numpy as np
from PIL import Image
import mlflow.pytorch
from model import RNAEncoder, ImageEncoder, CLIPModel

# Dataset definition
class RNACustomDataset(Dataset):
    def __init__(self, rna_data, targets, image_filenames, image_directory, transform=None):
        self.rna_data = rna_data.reset_index(drop=True)
        self.targets = targets.reset_index(drop=True)
        self.image_filenames = image_filenames
        self.image_directory = image_directory
        self.transform = transform

        self.valid_indices = []
        for i, fname in enumerate(self.image_filenames):
            if os.path.exists(os.path.join(self.image_directory, fname)):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        true_idx = self.valid_indices[idx]
        rna_sample = torch.tensor(self.rna_data.iloc[true_idx].values.astype(float)).float()
        target = torch.tensor(self.targets.iloc[true_idx]).float()

        img_path = os.path.join(self.image_directory, self.image_filenames[true_idx])
        nii_image = nib.load(img_path)
        image_volume = nii_image.get_fdata()
        mid_slice = image_volume[:, :, image_volume.shape[2] // 2]
        rgb_slice = np.stack([mid_slice] * 3, axis=-1)
        image = Image.fromarray(np.uint8(rgb_slice))

        if self.transform:
            image = self.transform(image)

        return rna_sample, image, target

# Load RNA-Seq Data (full 22126 genes)
rna_path = "C:/Users/95602/Desktop/ultrasound-contrastivemodels-klr-us-rna/TCIA/TCIA/GSE103584_R01_NSCLC_RNAseq.txt"
rna_df = pd.read_csv(rna_path, sep='\t', index_col=0).transpose()
rna_df = rna_df.fillna(0)

# Load clinical table with binary PDL1 and match sample IDs
excel_path = 'C:/Users/95602/Desktop/ultrasound-contrastivemodels-klr-us-rna/TCIA/TCIA/TCIA-cohorts1.xlsx'
df_clinical = pd.read_excel(excel_path)
df_clinical['Patient.ID'] = df_clinical['Patient.ID'].astype(str).str.strip()
rna_df.index = rna_df.index.str.strip()

matched_ids = []
rna_filtered = []
targets = []
image_filenames = []
image_directory = 'C:/Users/95602/Desktop/ultrasound-contrastivemodels-klr-us-rna/TCIA/TCIA/NSCLC Radiogenomics/NII'

for i, row in df_clinical.iterrows():
    pid = row['Patient.ID']
    try:
        label = int(row['PDL1'])
        if pid in rna_df.index and label in [0, 1]:
            image_path = os.path.join(image_directory, f"{pid}.nii.gz")
            if os.path.exists(image_path):
                sample = rna_df.loc[pid]
                matched_ids.append(pid)
                rna_filtered.append(sample)
                targets.append(label)
                image_filenames.append(f"{pid}.nii.gz")
    except:
        continue

rna_filtered_df = pd.DataFrame(rna_filtered)
targets_series = pd.Series(targets)

# Train-test split
rna_train, rna_test, y_train, y_test, img_train, img_test = train_test_split(
    rna_filtered_df, targets_series, image_filenames, test_size=0.2, random_state=42
)

# Data loaders
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_dataset = RNACustomDataset(rna_test, y_test, img_test, image_directory, transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIPModel(
    RNAEncoder(input_dim=22126),
    ImageEncoder()
).to(device)

model_uri = "file:///C:/Users/95602/Desktop/ultrasound-contrastivemodels-klr-us-rna/mlruns/1/2/artifacts/clip_model"
model.load_state_dict(mlflow.pytorch.load_model(model_uri).state_dict())
model.eval()

# Prediction head
class PD1PredictionHead(nn.Module):
    def __init__(self, embedding_dim=512):
        super(PD1PredictionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, rna_emb, img_emb):
        combined = torch.cat((rna_emb, img_emb), dim=1)
        return self.fc(combined)

predictor = PD1PredictionHead().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)

# Training loop with evaluation
for epoch in range(10):
    predictor.train()
    total_loss = 0
    for rna, images, targets in test_loader:
        rna, images, targets = rna.to(device), images.to(device), targets.to(device).float().unsqueeze(1)

        if torch.any(torch.isnan(targets)) or torch.any(targets < 0) or torch.any(targets > 1):
            print("warning")
            continue

        with torch.no_grad():
            rna_emb, img_emb = model(rna, images)
        optimizer.zero_grad()
        predictions = predictor(rna_emb, img_emb)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# Evaluation
predictor.eval()
y_true, y_prob = [], []
with torch.no_grad():
    for rna, images, targets in test_loader:
        rna, images = rna.to(device), images.to(device)
        rna_emb, img_emb = model(rna, images)
        probs = predictor(rna_emb, img_emb).cpu().numpy().flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(targets.numpy().tolist())

preds = [1 if p >= 0.5 else 0 for p in y_prob]
print("Accuracy:", accuracy_score(y_true, preds))
print("F1 Score:", f1_score(y_true, preds))
print("ROC AUC:", roc_auc_score(y_true, y_prob))
