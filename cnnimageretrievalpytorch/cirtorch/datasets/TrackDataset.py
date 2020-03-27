# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:00:15 2020

@author: joser
"""

import csv
import glob
import os
import numpy as np
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from cnnimageretrievalpytorch.cirtorch.datasets.datahelpers import imresize, pil_loader
from torchvision import transforms



class TrackDataset(Dataset):
    
    
    
    def __init__(self, set_size, is_train_set, transform=None, cnt_sampler = 10, PosEx = "Sequential", NegEx = "DiffLabel",
                 ResizeImage=(200,200)):
        '''
        set_size: number between 0 and 1 that represent the lenght of set
        is_train_set: bool that say if set is use for train or evaluation
        PosEx: Exist two option for the election of Positive example. One of them is "Random"
                where the example is choose from a ramdom election between over all option of frames of the
                same class. And the second option is "Sequential" that sample the positive example like a sequential of frames.
        
        NegEx: Variable for the election of negative example. Exist two option. "DiffLabel" use other class for the election of example.
                "Background" the election of example take the background of the same image of anchor.
        '''
        
        
        
        path = "D:/matricula u chile 2015/12 semestre/Imagenes_avanzado/Proyecto/Reconocedor/ProyectoYoloSPoC/NewDirDB/TLP"
        self.Dataset_list = []
        self.grt_list = []
        self.label = []
        self.transform = transform
        self.PosEx = PosEx
        self.NegEx = NegEx
        self.ResizeImage = ResizeImage
        numOfFolder = 0
        le = preprocessing.LabelEncoder()
#        FrameDisc = []
        
        for folder_obj in glob.glob(os.path.join(path,'*')):
            
            if numOfFolder > 9:
                break
            
            f = open(os.path.join(folder_obj,'groundtruth_rect.txt'), 'r')
            data = f.readlines()
            reader = csv.reader(data)
            frames_per_folder = sum(1 for row in reader)
            f.seek(0)
            reader = csv.reader(data)
            f.close()
            if is_train_set:
                limit_idx = int(frames_per_folder*set_size)
            else:
                limit_idx = int(frames_per_folder*(1-set_size))
            
#            for row in reader:
#                self.grt_list.append(row)
#            self.grt_list = [row for idx, row in enumerate(reader) if idx%cnt_sampler==0 and row[-1] == 0] # Objeto Aparece en la imagen
            for idx, row in enumerate(reader):
#                if idx%cnt_sampler == 0 and row[-1] == '0': # Object must be on screen
                if is_train_set:
                    if idx < limit_idx and idx%cnt_sampler==0:
                    
                        self.grt_list.append(row)
                else:
                    if idx >= limit_idx and idx%cnt_sampler==0:
                        self.grt_list.append(row)
                        
#                elif row[-1] == 1:
#                    FrameDisc.append(idx)
            cnt = 0
            for filename in glob.glob(os.path.join(folder_obj,'img','*jpg')):
#                if cnt%cnt_sampler==0 and cnt not in FrameDisc:
                if is_train_set:
                    if cnt < limit_idx and cnt%cnt_sampler==0:
                    
                        self.label.append(folder_obj.split('P')[-1][1:])
                        self.Dataset_list.append(filename)
                else:
                    if cnt >= limit_idx and cnt%cnt_sampler==0:
                        self.label.append(folder_obj.split('P')[-1][1:])
                        self.Dataset_list.append(filename)
                cnt += 1
                
            numOfFolder += 1
        
        
        self.label = le.fit_transform(self.label)
    
    def __getitem__(self, index):
        
        filename_anchor = self.Dataset_list[index]
        anchor_img = pil_loader(filename_anchor) # Todas estan con las mismas dimensiones
        anchor_label = self.label[index]
        anchor_grt = self.grt_list[index]
        anchor_grt = [int(val) for val in anchor_grt]
        anchor_img = anchor_img.crop((anchor_grt[1], anchor_grt[2], anchor_grt[1] + anchor_grt[3],
                        anchor_grt[4]  + anchor_grt[2]))
        

        positive_list = [i for i, j in enumerate(self.label) if j==anchor_label]
        positive_idx = index
        
        if self.PosEx == "Random":
            while positive_idx == index:
                positive_idx = np.random.choice(positive_list)
        elif self.PosEx == "Seq":
            if filename_anchor == positive_list[-1]: # Special case for the last frame
                positive_idx = index - 1
            else:
                positive_idx = index + 1 
        positive_img = pil_loader(self.Dataset_list[positive_idx])
        positive_label = self.label[positive_idx]
        positive_grt = self.grt_list[positive_idx]
        positive_grt = [int(val) for val in positive_grt]
        positive_img = positive_img.crop((positive_grt[1], positive_grt[2], positive_grt[1] + positive_grt[3],
                        positive_grt[4] + positive_grt[2]))
        if self.NegEx == "DiffLabel":
            negative_list = [i for i, j in enumerate(self.label) if j != anchor_label]
            negative_idx = np.random.choice(negative_list)
            negative_img = pil_loader(self.Dataset_list[negative_idx])
            negative_label = self.label[negative_idx]
            negative_grt = self.grt_list[negative_idx]
            negative_grt = [int(val) for val in negative_grt]
            negative_img = negative_img.crop((negative_grt[1], negative_grt[2], negative_grt[1] + negative_grt[3],
                            negative_grt[4] + negative_grt[2]))
        
        elif self.NegEx == "Background":
            negative_idx = index
            negative_img = pil_loader(self.Dataset_list[negative_idx])
            
            
        anchor_img = anchor_img.resize(self.ResizeImage)
        positive_img = positive_img.resize(self.ResizeImage)
        negative_img = negative_img.resize(self.ResizeImage)
        
#        display(anchor_img)
#        display(positive_img)
#        display(negative_img)
#        print(self.Dataset_list[negative_idx])
        
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        targets = torch.as_tensor([anchor_label, positive_label, negative_label])
    
#        targets = [anchor_label, positive_label, negative_label]
        return ([anchor_img, positive_img, negative_img], targets)
        
        
        
    def __len__(self):
        return len(self.Dataset_list)
        
if __name__ == '__main__':
    
    transform = transforms.Compose([transforms.ToTensor()])
    params = {'batch_size': 1}
    training_set = TrackDataset(0.9, True, transform = transform, PosEx="Seq")
    validation_set = TrackDataset(0.1, False, transform = transform, PosEx="Seq")
    
    training_generator = DataLoader(training_set, **params)
    validation_generator = DataLoader(validation_set, **params)
    
    cnt = 0
    for imgs, labels in training_generator:
        
#        print(labels.numpy()[:,0])
        cnt += 1
        
        