import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from utils.utils import compute_embeddings

def retrieve_knn(query_embeddings, db_embeddings, k, device) -> np.array :
    """
    Retrieve top-k nearest neighbors using cosine similarity
    
    Args:
        query_embeddings: Embeddings of query images        [num_queries, 2048]
        db_embeddings: Embeddings of images in database     [num_db, 2048]
        k: number of nearest neighbors to retrieve.
        
    Returns:
        indices: Indices of top-k nearest neighbors for each query 
    """
    
    query_embeddings = query_embeddings
    db_embeddings = db_embeddings.to(device)
    
    # L2 Norm
    query_embeddings = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
    db_embeddings = db_embeddings / db_embeddings.norm(dim=1, keepdim=True)
    
    dataset = TensorDataset(query_embeddings)
    loader = DataLoader(dataset, 
                        batch_size=128, 
                        num_workers=8,
                        pin_memory=True,
                        shuffle=False)

    indices = []
    
    for query_batch in tqdm(loader, desc='Computing Similarities'):
        query_batch = query_batch[0].to(device)
        similarity_scores = torch.matmul(query_batch, db_embeddings.T)                          # [batch_size, embed_dim] x [embed_dim, num_db]
        _, topk_indices = torch.topk(similarity_scores, k=k, dim=1, largest=True, sorted=True)  # [batch_size, k]
        indices.append(topk_indices)      
        
    indices = torch.concat(indices, dim=0)
    return indices.cpu().numpy()


def calculate_recall_at_k(indices, query_labels, db_labels, k):
    """
    Calculate Recall@k for the given retrieval results.
    
    Args:
        indices: Indices of top-k nearest neighbors.
        query_labels: Ground-truth labels of the query set.
        db_labels: Ground-truth labels of the database set.
        k: Number of neighbors considered.
        
    Returns:
        recall: Recall@k score(%)
    """
    num_queries = len(query_labels)
    correct_cnt = 0
    
    for i in tqdm(range(num_queries)):
        retrieved_labels = db_labels[indices[i, :k]]
        if query_labels[i] in retrieved_labels:
            correct_cnt += 1
    
    recall = (correct_cnt / num_queries) * 100
    return recall


def evaluate_recall(model, query_loader, db_loader, k_values, device):
    """
    Evaluate Recall@k for a given model and dataset
    
    Args:
        model: Trained model.
        query_loader: DataLoader for query dataset.
        db_loader: DataLoader for database dataset.
        k_values: List of k values for Recall@k.
        device: Device to run the evaluation
        
    Returns:
        recall_scores: Dictionaty with Recall@k values.
    """
    
    query_embeddings, query_labels, _ = compute_embeddings(model, query_loader, device)
    db_embeddings, db_labels, _ = compute_embeddings(model, db_loader, device)
    
    max_k = max(k_values)
    indices = retrieve_knn(query_embeddings, db_embeddings, max_k, device)
    
    recall_scores = {}
    for k in k_values:
        recall = calculate_recall_at_k(indices, query_labels, db_labels, k)
        recall_scores[f'Recall@{k}'] = recall
        
    return recall_scores