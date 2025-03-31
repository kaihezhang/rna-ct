import os
import torch
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import torch.nn as nn
import sys

# 添加 src 路径到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import CLIPModel, RNAEncoder, ImageEncoder, ContrastiveLoss
from dataset import RNACustomDataset

# Set MLflow tracking path
mlflow.set_tracking_uri("file:///C:/Users/95602/Desktop/ultrasound-contrastivemodels-klr-us-rna/mlruns")

# RNA-Seq data path 和 图像路径
rna_path = "C:/Users/95602/Desktop/ultrasound-contrastivemodels-klr-us-rna/TCIA/TCIA/GSE103584_R01_NSCLC_RNAseq.txt"
image_directory = "C:/Users/95602/Desktop/ultrasound-contrastivemodels-klr-us-rna/TCIA/TCIA/NSCLC Radiogenomics/NII"

# 步骤1: 读取 RNA 表达量数据
# RNA 数据的列是 sample ID，首行是 gene symbol
rna_df = pd.read_csv(rna_path, sep='\t', index_col=0)
rna_df = rna_df.transpose()  # 转置以便根据 sample ID 查询

# 步骤2: 匹配 RNA 和图像
image_filenames = []
rna_data_filtered = []
pair_count = 0
max_pairs = 100

for sample_id in rna_df.index:
    if pair_count >= max_pairs:
        break
    image_filename = f"{sample_id}.nii.gz"
    image_path = os.path.join(image_directory, image_filename)
    if os.path.exists(image_path):
        image_filenames.append(image_filename)
        row = rna_df.loc[sample_id].apply(pd.to_numeric, errors='coerce').fillna(0)
        rna_data_filtered.append(row)
        pair_count += 1

# 步骤3: 转换成 DataFrame
rna_data_filtered = pd.DataFrame(rna_data_filtered)

# 分割数据
rna_train, rna_test, img_train, img_test = train_test_split(
    rna_data_filtered, image_filenames, test_size=0.2, random_state=42
)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = RNACustomDataset(rna_train, img_train, image_directory, transform=transform)
test_dataset = RNACustomDataset(rna_test, img_test, image_directory, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 模型 + 传感器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rna_encoder = RNAEncoder(input_dim=rna_data_filtered.shape[1], embedding_dim=512)
image_encoder = ImageEncoder(embedding_dim=512)
model = CLIPModel(rna_encoder, image_encoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = ContrastiveLoss(temperature=0.5)

# MLflow
mlflow.set_experiment("RNA-Image CLIP Model")

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    with mlflow.start_run():
        mlflow.log_param("learning_rate", 1e-4)
        mlflow.log_param("batch_size", 4)
        mlflow.log_param("embedding_dim", 512)

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for rna_batch, image_batch in train_loader:
                rna_batch, image_batch = rna_batch.to(device), image_batch.to(device)
                rna_embeddings, image_embeddings = model(rna_batch, image_batch)
                loss = criterion(rna_embeddings, image_embeddings)
                total_train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_train_loss = total_train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for rna_batch, image_batch in val_loader:
                    rna_batch, image_batch = rna_batch.to(device), image_batch.to(device)
                    rna_embeddings, image_embeddings = model(rna_batch, image_batch)
                    loss = criterion(rna_embeddings, image_embeddings)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        mlflow.pytorch.log_model(model, "clip_model")

# 训练
train(model, train_loader, test_loader, optimizer, criterion, device, epochs=10)
