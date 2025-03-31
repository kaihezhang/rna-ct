import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class RNAEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=512):
        super(RNAEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        return self.model(x)


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.model = resnet

    def forward(self, x):
        return self.model(x)


class CLIPModel(nn.Module):
    def __init__(self, rna_encoder, image_encoder):
        super(CLIPModel, self).__init__()
        self.rna_encoder = rna_encoder
        self.image_encoder = image_encoder

    def forward(self, rna_input, image_input):
        rna_embeddings = self.rna_encoder(rna_input)
        image_embeddings = self.image_encoder(image_input)
        return rna_embeddings, image_embeddings


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, rna_embeddings, image_embeddings):
        rna_embeddings = nn.functional.normalize(rna_embeddings, p=2, dim=-1)
        image_embeddings = nn.functional.normalize(image_embeddings, p=2, dim=-1)
        similarity_matrix = self.cosine_similarity(rna_embeddings.unsqueeze(1),
                                                   image_embeddings.unsqueeze(0)) / self.temperature
        labels = torch.arange(rna_embeddings.size(0)).to(rna_embeddings.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = (loss_fn(similarity_matrix, labels) + loss_fn(similarity_matrix.t(), labels)) / 2
        return loss
