import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class GenericDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        keypoints = sample['keypoints']
        labels = sample['labels']

        if self.transform:
            keypoints = self.transform(keypoints)

        return {'keypoints': torch.tensor(keypoints, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long)}

def get_data_loaders(dataset_name, batch_size=32, shuffle=True, num_workers=4, transform=None):
    data_path = os.path.join('data', f'{dataset_name}_processed_data.pkl')
    dataset = GenericDataset(data_path, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

if __name__ == '__main__':
    dataset_name = 'Penn_Action'
    data_loader = get_data_loaders(dataset_name)

    for batch in data_loader:
        print(batch['keypoints'].shape, batch['labels'].shape)
        break
