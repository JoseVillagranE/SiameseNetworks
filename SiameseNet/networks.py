# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:19:26 2020

@author: joser
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, TypeOfNet='AlexNet', normalization=False):
        super().__init__()
        
        self.TypeOfNet = TypeOfNet
        self.normalization = normalization
        if(TypeOfNet=='AlexNet'):
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True), 
                                         nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                        )
            
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1000) # num_classes = 1000 -> DB ImageNet
            )
        
        elif(TypeOfNet == 'Small'):
        
            self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(), 
                                         nn.MaxPool2d(2, stride=2), 
                                         nn.Conv2d(32, 64, 5), nn.PReLU(), 
                                         nn.MaxPool2d(2, stride=2))
            self.fc = nn.Sequential(nn.Linear(64*47*47, 64),
                                    nn.PReLU(),
                                    nn.Linear(64, 32),
                                    nn.PReLU(),
                                    nn.Linear(32, 2))
        
        
    def forward(self, x):
        
        if(self.TypeOfNet=='AlexNet'):
            out = self.features(x)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.classifier(out)
        elif(self.TypeOfNet=='Small'):
            out = self.convnet(x)
            out = out.view(out.size()[0],-1)
            out = self.fc(out)
        
        if(self.normalization):
            out = F.normalize(out, dim=0, p=2) # L2 normalization
        
        return out
    
    def get_embedding(self, x):
        return self.forward(x)
        
        

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embeddingNet = embedding_net
        
    def forward(self, x1, x2, x3):
        out1 = self.embeddingNet(x1)
        out2 = self.embeddingNet(x2)
        out3 = self.embeddingNet(x3)
        return out1, out2, out3
    
    def get_embedding(self, x):
        return self.embeddingNet(x)
        
    
        
        