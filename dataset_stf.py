import torch.utils.data as data
import torch
import h5py
import random
import numpy as np

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path,data_enhanced):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path,'r')
        self.mod1 = hf.get('mod1')
        self.mod2 = hf.get('mod2')
        self.land1 = hf.get('land1')
        self.land2 = hf.get('land2')
        self.data_enhanced = data_enhanced


    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.data_enhanced:
            mod1 = self.arguement(self.mod1[index,:,:,:], rotTimes, vFlip, hFlip)
            mod2 = self.arguement(self.mod2[index, :, :, :], rotTimes, vFlip, hFlip)
            land1 = self.arguement(self.land1[index, :, :, :], rotTimes, vFlip, hFlip)
            land2 = self.arguement(self.land2[index, :, :, :], rotTimes, vFlip, hFlip)
        else:
            mod1 =self.mod1[index, :, :, :]
            mod2 = self.mod2[index, :, :, :]
            land1 = self.land1[index, :, :, :]
            land2 = self.land2[index, :, :, :]

        return torch.from_numpy(mod1.copy()).float(),torch.from_numpy(mod2.copy()).float(),torch.from_numpy(land1.copy()).float(),torch.from_numpy(land2.copy()).float()
        
    def __len__(self):
        return self.land2.shape[0]