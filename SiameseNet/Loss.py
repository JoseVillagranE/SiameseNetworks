# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:11:52 2020

@author: joser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Constrative Loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same
    class and label == 0 otherwise
    """
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9
        
    def forward(self, out1, out2, target, size_average=True):
        distances = (out2 - out1).pow(2).sum(1)
        losses = 0.5*(target.float()*distances + (1 - target).float()*F.relu(self.margin - 
                      (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()



class TripletLoss(nn.Module):
    
    """
    Triplet Loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        
    
    def forward(self, anchor, positive, negative, size_averange=True):
        
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_averange else losses.sum()
        
    
class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and
    targets and return indices of triplets
    """
    
    def __init__(self, margin, triplet_selector):
        super().__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        
    
    def forward(self, embeddings, target):
        
        triplets = self.triplet_selector.get_triplets_seq(embeddings, target)
        if embeddings[0].is_cuda:
            triplets = triplets.cuda()
        
        if(len(triplets)>0):
            ap_distances = (embeddings[0][triplets[:,0]] - embeddings[1][triplets[:,1]]).pow(2).sum(1)
            an_distances = (embeddings[0][triplets[:,0]] - embeddings[2][triplets[:,2]]).pow(2).sum(1)
            losses = F.relu(ap_distances - an_distances + self.margin)
            return losses.mean(), len(triplets)
        else:
            return 0, 0