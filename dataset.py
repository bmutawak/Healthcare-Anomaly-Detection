import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SignalDataset(Dataset):
    
    def __init__(self, data, seqLen, normalize=False, means=None, stdevs=None, num_train=None):
        """
        Initializes signal dataset. Portions the given data into sequences
        of seqLen.

        Parameters
        ----------
        data : ndarray
            Input dataset.
        seqLen : int
            Length of the sequence to cut.

        Returns
        -------
        None.

        """
        
        
        # Partition the data into sequences
        numRemove = int(data.shape[0] % seqLen)
        if not numRemove == 0:
            data = np.copy(data)[:-1*numRemove, :]
        else:
            data = np.copy(data)
            
        numInstances, self.numFeatures = data.shape
        self.numSequences = numInstances // seqLen
        self.normalize = normalize
        self.means = means
        self.stdevs = stdevs
        self.num_train = num_train
        self.seqLen = seqLen
        
        # Create a holder for the sequences
        data_holder = np.zeros((seqLen, self.numFeatures, self.numSequences))
        prev_index = 0
        for i in range(0, numInstances, seqLen):
            data_holder[:, :, i // seqLen] = data[prev_index: (i + seqLen), :]
            prev_index = i + seqLen
        
        self.data = data_holder
        return
    
        
    def __len__(self):
        return self.numSequences
    
    def normalizeData(self, data, idx):
        # Normalize and update means, stdevs
        iteration = self.num_train + idx
        for idx in range(data.shape[0]):
            
            for i in range(self.means.shape[-1]):
                new_mean = self.means[i] + ((data[idx, i] - self.means[i]) / iteration)
                variance = self.stdevs[i] ** 2
                self.stdevs[i] = np.sqrt((1 / (1 + iteration)) * (iteration * variance + (data[idx, i] - new_mean) * (data[idx, i] - new_mean)))
                self.means[i] = new_mean

    
            data[idx, :] = (data[idx, :] - self.means) / self.stdevs
            
        return data
    
    def __getitem__(self, idx):
        # Read file at given index
        data = self.data[:, :, idx].reshape((self.seqLen, self.numFeatures))
        
        if self.normalize:
            data = self.normalizeData(data, idx)
        
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        return data
    
    
    
def get_dataloaders(train_set, test_set, val_set, means, stdevs, num_train, normalize_online = True, seqLen = 4, batch_size=8):
    
    # Create datasets
    train_dataset = SignalDataset(train_set, seqLen, normalize=False)
    test_dataset = SignalDataset(test_set, seqLen, normalize=False)
    val_dataset = SignalDataset(val_set, seqLen, normalize=normalize_online, means=means, stdevs=stdevs, num_train=num_train)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, 
                              pin_memory=True, num_workers=8)
    
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, 
                              pin_memory=True, num_workers=8)
    
    val_loader = DataLoader(val_dataset, batch_size=1, 
                              pin_memory=True, num_workers=8)
    return train_loader, test_loader, val_loader
    
    
    
    
    
    
    
    
    
    
    
    
    