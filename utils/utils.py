import torch

from sklearn.manifold import TSNE
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm


def compute_embeddings(model, loader, device) -> torch.Tensor:
    """
    Compute Embeddings for a given model and dataset
    
    Args:
        dataloader: DataLoader for query dataset.
        model: Trained model.
        device: Device to run the computing emebddings
        
    Returns:
        all_embeddings: Embedding vector for all query images
        all_labels: Labels for images in whole dataset
        all_super_labels: Super labels for images whole dataset
    """
    
    all_embeddings = []
    all_labels = []
    all_super_labels = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc='Extracting embeddings'):
            images = data['image'].to(device)
            labels = data['label'].to(device)
            super_labels = data['super_label'].to(device)
            
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_super_labels.append(super_labels.cpu())
            
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_super_labels = torch.cat(all_super_labels, dim=0)
    
    return all_embeddings, all_labels, all_super_labels


def plot_tsne(model, loader, device, sample_size=1000, title='t-SNE Visualization') -> None:
    """
    Plot T-SNE visualization results
    
    Args:
        embeddings: Embeddings vectors to be visualized with T-SNE
        labels: Labels for given samples
        sample_size: Number of samples to be visualized with T-SNE
        title: Title for T-SNE figure
    """
    embeddings, _, super_labels = compute_embeddings(model, loader, device)
    embeddings_sampled, super_labels_sampled = shuffle(embeddings, super_labels, n_samples=sample_size)
    
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings_sampled)
    
    unique_labels = np.unique(super_labels_sampled)
    plt.figure(figsize=(10,10))
    
    for label in unique_labels:
        indices = np.where(super_labels_sampled == label)
        plt.scatter(
            reduced_embeddings[indices, 0],
            reduced_embeddings[indices, 1],
            label = f"Label {label}",
            alpha=0.7
        )
        
    plt.title(title)
    plt.legend()
    plt.show()
    

# def create_triplets_balanced_super_label(labels, super_labels):
#     """
#     labels: (batch_size, 1) - 샘플별 라벨
#     super_labels: (batch_size, 1) - 샘플별 상위 라벨
#     """
#     labels = labels.squeeze()  # 1D로 변환
#     super_labels = super_labels.squeeze()  # 1D로 변환

#     # super_label 그룹 분리
#     unique_super_labels = torch.unique(super_labels)
#     assert len(unique_super_labels) == 2, "Batch must have exactly 2 super_labels for balanced splitting"

#     group1_indices = torch.where(super_labels == unique_super_labels[0])[0]
#     group2_indices = torch.where(super_labels == unique_super_labels[1])[0]

#     triplets = []

#     # 같은 그룹에서 Anchor-Positive 구성
#     for group, other_group in [(group1_indices, group2_indices), (group2_indices, group1_indices)]:
#         for i in range(len(group)):
#             anchor_idx = group[i]
#             # Positive는 같은 그룹 내에서 선택
#             positive_idx = group[(i + 1) % len(group)]  # 순환적으로 선택

#             # Negative는 다른 그룹에서 선택
#             negative_idx = other_group[torch.randint(0, len(other_group), (1,)).item()]

#             triplets.append((anchor_idx.item(), positive_idx.item(), negative_idx.item()))

#     return triplets
    
            