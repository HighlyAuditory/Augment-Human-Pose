import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch
from . import debugger

import pdb

class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints
        self.dir_json = os.path.join(opt.dataroot, opt.phase + '_3d_top_ordered') #keypoints

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1
        K1_path = os.path.join(self.dir_json, P1_name.replace('.jpg','.npy'))

        # person 2 and its bone 
        P2_path = os.path.join(self.dir_P, P2_name) # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2
        K2_path = os.path.join(self.dir_json, P2_name.replace('.jpg','.npy'))

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path) 

        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0,1)
            
            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip

            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        else:
            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        Kd1, Kd2 = np.load(K1_path,allow_pickle=True).item(), np.load(K2_path,allow_pickle=True).item()

        # should expand dims

        K1, K2 = Kd1['absolute_angles'], Kd2['absolute_angles']
        L2, F2 = Kd2['limbs'], Kd2['offset'].squeeze()
        L1, F1 = Kd1['limbs'], Kd1['offset'].squeeze()

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name, 'K1': K1, 'K2': K2,
                'L2':L2, 'F2': F2, 'L1':L1, 'F1': F1}
                

    def __len__(self):
        if self.opt.phase == 'train':
            return 1008
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'
