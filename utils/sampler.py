import torch
import itertools 
import numpy as np           
import random 
    
class Sampler():
    def __init__(self, method='ramdom'):
        
        if method == 'random':
            self.give = self.randomsampling
        elif method == 'semihard':
            self.give = self.semihardsampling
        elif method == 'distance':
            self.give = self.distanceweightedsampling
        else:
            raise NotImplemented
        
    def randomsampling(self, batch, labels):
        if isinstance(labels, torch.Tensor): 
            labels = labels.detach().cpu().numpy()
        unique_classes = np.unique(labels)
        indices = np.arange(len(batch))
        class_dict = {i:indices[labels.squeeze() == i] for i in unique_classes}
        
        sampled_triplets = [list(itertools.product([x], [x], [y for y in unique_classes if x != y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]
        
        sampled_triplets = [[x for x in list(itertools.product(*[class_dict[j] for j in i])) if x[0] != x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]
        
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
        
        return sampled_triplets
    
    def semihardsampling(self, batch, labels, margin=0.2):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels!=l; pos = labels==l

            anchors.append(i)
            pos[i] = False
            p = np.random.choice(np.where(pos)[0])
            positives.append(p)

            #Find negatives that violate tripet constraint semi-negatives
            neg_mask = np.logical_and(neg,d>d[p])
            neg_mask = np.logical_and(neg_mask,d<margin+d[p])
            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets
    
    
    def distanceweightedsampling(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on distance sampling introduced in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
            lower_cutoff: float, lower cutoff value for negatives that are too close to anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
            upper_cutoff: float, upper cutoff value for positives that are too far away from the anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        sampled_triplets = [[a,p,n] for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets
    
    def pdist(self, A):
        """
        Efficient function to compute the distance matrix for a matrix A.

        Args:
            A:   Matrix/Tensor for which the distance matrix is to be computed.
            eps: float, minimal distance/clampling value to ensure no zero values.
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = 0).sqrt()
    
    def inverse_sphere_distances(self, batch, dist, labels, anchor_label):
        """
        Function to utilise the distances of batch samples to compute their
        probability of occurence, and using the inverse to sample actual negatives to the resp. anchor.

        Args:
            batch:        torch.Tensor(), batch for which the sampling probabilities w.r.t to the anchor are computed. Used only to extract the shape.
            dist:         torch.Tensor(), computed distances between anchor to all batch samples.
            labels:       np.ndarray, labels for each sample for which distances were computed in dist.
            anchor_label: float, anchor label
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        bs,dim       = len(dist),batch.shape[-1]

        #negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
        #Set sampling probabilities of positives to zero
        log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

        q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
        #Set sampling probabilities of positives to zero
        q_d_inv[np.where(labels==anchor_label)[0]] = 0

        ### NOTE: Cutting of values with high distances made the results slightly worse.
        # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

        #Normalize inverted distance for probability distr.
        q_d_inv = q_d_inv/q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()
    
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='random'):
        """
        Basic Triplet Loss as proposed in 'FaceNet: A Unified Embedding for Face Recognition and Clustering'
        Args:
            margin:             float, Triplet Margin - Ensures that positives aren't placed arbitrarily close to the anchor.
                                Similarl, negatives should not be placed arbitrarily far away.
            sampling_method:    Method to use for sampling training triplets. Used for the TupleSampler-class.
        """
        super(TripletLoss, self).__init__()
        self.margin             = margin
        self.sampler            = Sampler(method=sampling_method)

    def triplet_distance(self, anchor, positive, negative):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            triplet loss (torch.Tensor(), batch-averaged)
        """
        #Sample triplets to use for training.
        sampled_triplets = self.sampler.give(batch, labels)
        #Compute triplet loss
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return torch.mean(loss)    

class MarginLoss(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='random'):
        """
        Basic Margin Loss as proposed in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            margin:          float, fixed triplet margin (see also TripletLoss).
            nu:              float, regularisation weight for beta. Zero by default (in literature as well).
            beta:            float, initial value for trainable class margins. Set to default literature value.
            n_classes:       int, number of target class. Required because it dictates the number of trainable class margins.
            beta_constant:   bool, set to True if betas should not be trained.
            sampling_method: str, sampling method to use to generate training triplets.
        Returns:
            Nothing!
        """
        super(MarginLoss, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant

        self.beta_val = beta
        self.beta     = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu = nu

        self.sampling_method    = sampling_method
        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            margin loss (torch.Tensor(), batch-averaged)
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        sampled_triplets = self.sampler.give(batch, labels)

        #Compute distances between anchor-positive and anchor-negative.
        d_ap, d_an = [],[]
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

            pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
            neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)
 
        #Group betas together by anchor class in sampled triplets (as each beta belongs to one class).
        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        #Compute actual margin postive and margin negative loss
        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        #Compute normalization constant
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        #Actual Margin Loss
        loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count

        #(Optional) Add regularization penalty on betas.
        # if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss