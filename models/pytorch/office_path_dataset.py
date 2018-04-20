from torch.utils.data import Dataset

class OfficePathDataset(Dataset):

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
