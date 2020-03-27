# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:15:06 2020

@author: joser
"""

from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def pdist(vectors):
    distance_matrix = (-2*vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1,-1) + 
                        vectors.pow(2).sum(dim=1).view(-1,1))
    
    return distance_matrix

def pdist_2(vectors):
    anchor, pos, neg = vectors
    
    anchor_norm = anchor.pow(2).sum(1).view(-1, 1)
    
    negative_norm = neg.pow(2).sum(1).view(1, -1)
    
    an_distance_matrix = anchor_norm + negative_norm - 2*torch.mm(anchor, torch.transpose(neg, 0, 1))
    
    ap_distance = F.pairwise_distance(anchor, pos)
    
    return an_distance_matrix, ap_distance

    
def extract_embeddings(dataloader, model, cuda):
    
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for data, target in dataloader:
            if cuda:
                data = data[0].cuda()
            embeddings[k:k+len(data)] = model.get_embedding(data).data.cpu().numpy()
            labels[k:k+len(data)] = target.numpy()[:,0]
            k += len(data)
    
    return embeddings, labels

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    
    plt.figure(figsize=(10,10))
    labels = np.unique(targets)
    for i in range(labels.shape[0]):
        idxs = np.where(targets==i)[0]
        plt.scatter(embeddings[idxs, 0], embeddings[idxs, 1], alpha=0.5)
    plt.legend(labels)

def SilhoutteCoeff(embeddings, targets, newFig=True):
    
#    targets = np.array([int(i) for i in targets])
    
    
#    if(newFig):
#        
#        plt.figure(figsize=(10,10))
#        
    labels = np.unique(targets)
    S = []
    for i in range(labels.shape[0]):
        idxs = np.where(targets==i)[0]
        labels_compl = np.where(labels!=i)[0]
        DistanceMat = np.zeros((idxs.shape[0], idxs.shape[0])) # Distances intra-cluster
        b = []
        for j in range(idxs.shape[0] - 1):
            pj = embeddings[idxs[j], :] # actual point
            pk = embeddings[idxs[j+1:], :]
            distances = np.linalg.norm(pk-pj, axis=1)
            DistanceMat[j, j+1:] = distances
            DistanceMat[j+1:, j] = distances
            listDistanceOther = []
            for k in labels_compl:
                
                idxs_otherClus = np.where(targets == k)[0]
                pk_other = embeddings[idxs_otherClus, :]
                distances_other = np.linalg.norm(pk_other-pj, axis=1)
                distances_other = np.mean(distances_other)
                listDistanceOther.append(distances_other)
            b.append(min(listDistanceOther))
            
        pj = embeddings[idxs[-1], :]
        for k in labels_compl:
                
            idxs_otherClus = np.where(targets == k)[0]
            pk_other = embeddings[idxs_otherClus, :]
            distances_other = np.linalg.norm(pj - pk_other)
            listDistanceOther.append(distances_other)
        b.append(min(listDistanceOther))
        
        a = np.mean(DistanceMat, axis=1)
        b = np.array(b)       
        for z in range(a.shape[0]):
            S_i = (b[z] - a[z])/(max(a[z], b[z]))
            S.append(S_i)
    
#    plt.scatter(range(len(S)), S, s=1)
#    plt.plot(range(len(S)), S)
#    plt.xlabel("Image")
#    plt.ylabel("Silhouette Coeff.")
#    
#    plt.ylim(-1, 1)
    return S
    
        

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """
    
    def __init__(self):
        pass
    
    def get_triplets(self, embeddings, labels):
        raise NotImplementedError
        
        
def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None
#    return hard_negative

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunNegTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value)
    to create a triplet. Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor_positive pair
    and all negative samples and return a negative index for that pair
    """
    
    def __init__(self, margin, negative_selection_fn, cpu=True):
        super().__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        
        
    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()
        
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in labels:
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distances in zip(anchor_positives, ap_distances):
                loss_values = (ap_distances - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), 
                                                            torch.LongTensor(negative_indices)] +
                                                            self.margin)
            
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
                    
                
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
            
        triplets = np.array(triplets)
        
        return torch.LongTensor(triplets)
    
    def get_triplets_seq(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        an_distance_matrix, ap_distances = pdist_2(embeddings)
        an_distance_matrix = an_distance_matrix.cpu()
        ap_distances = ap_distances.cpu()
        labels = labels.cpu().data.numpy()
        triplets = []
        
        for i in range(an_distance_matrix.shape[0]):
            loss_values = []
            for j in range(an_distance_matrix.shape[1]):
                loss_value = (ap_distances[i] - an_distance_matrix[i, j] + self.margin)
                loss_value = loss_value.data.cpu().numpy()
                loss_values.append(loss_value)
            idx_hard_negative = self.negative_selection_fn(loss_values)
            if idx_hard_negative is not None:
#                hard_negative = embeddings[2][idx_hard_negative, 2]
                triplets.append([i, i, idx_hard_negative])
                
        triplets = np.array(triplets)
        return torch.LongTensor(triplets)
            
            
        
    
def HardestNegTripletSelector(margin, cpu=False):
    return FunNegTripletSelector(margin, hardest_negative, cpu=cpu)

def RandomNegTripletSelector(margin, cpu = False):
    return FunNegTripletSelector(margin, random_hard_negative, cpu = cpu)

def SemiHardNegTripletSelector(margin, cpu = False):
    return FunNegTripletSelector(margin, lambda x: semihard_negative(x, margin), cpu = cpu)


if __name__ == "__main__":
    
    embeddings = np.array([[1,2], [1,3], [1,4], [10, 2], [10, 3], [10, 4]])
    targets = np.array([0 , 0, 0, 1, 1, 1])

    S = SilhoutteCoeff(embeddings, targets)
    
    plt.figure(figsize=(10,10))
    plt.plot(range(len(S)), S)
    k = np.array([np.mean(S) for i in range(len(S))])
    plt.plot(range(len(S)), k, '--')
    plt.ylim(-1, 1)
    plt.xlabel("Image")
    plt.ylabel("Silhouette Coeff.")
    plt.show()
    