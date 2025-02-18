import torch.utils.data as data
import torch
import h5py
import random
import numpy as np

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path,data_enhanced):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path,'r')
        self.mul = hf.get('data1')
        self.pan = hf.get('data2')
        self.target = hf.get('label')
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
            mul= self.arguement(self.mul[index,:,:,:], rotTimes, vFlip, hFlip)
            pan = self.arguement(self.pan[index, :, :, :], rotTimes, vFlip, hFlip)
            target = self.arguement(self.target[index, :, :, :], rotTimes, vFlip, hFlip)
        else:
            mul = self.mul[index, :, :, :]
            pan = self.pan[index, :, :, :]
            target = self.target[index, :, :, :]

        return torch.from_numpy(mul.copy()).float(), torch.from_numpy(pan.copy()).float(), torch.from_numpy(target.copy()).float()
        
    def __len__(self):
        return self.mul.shape[0]