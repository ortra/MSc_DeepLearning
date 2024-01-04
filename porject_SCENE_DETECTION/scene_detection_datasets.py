import os
from PIL import Image
from torch.utils.data import Dataset

"""
script that contain the dataset classes that we used 
"""
def _sort_by_numerical_part(items):
    def extract_numerical_part(item):
        return int(''.join(filter(str.isdigit, item)))

    return sorted(items, key=extract_numerical_part)


def _sort_alphabetically(items):
    return sorted(items)


class CamSDD(Dataset):
    def __init__(self, root_path: str, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.categories = _sort_by_numerical_part(os.listdir(root_path))
        self.prompts = [category.split('_', 1)[1] for category in self.categories]
        self.index_mapping = self._create_index_mapping()

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        category, file_idx = self.index_mapping[idx]
        category_path = os.path.join(self.root_path, category)
        img_name = os.listdir(category_path)[file_idx]
        img_path = os.path.join(category_path, img_name)

        # Use PIL to open the image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.categories.index(category)

    def _create_index_mapping(self):
        index_mapping = []
        for category_idx, category in enumerate(self.categories):
            category_path = os.path.join(self.root_path, category)
            num_files = len(os.listdir(category_path))
            index_mapping.extend([(category, file_idx) for file_idx in range(num_files)])
        return index_mapping


class Places365(Dataset):
    def __init__(self, root_path: str, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.categories = _sort_alphabetically(os.listdir(root_path))
        self.prompts = self.categories
        self.index_mapping = self._create_index_mapping()

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        category, file_idx = self.index_mapping[idx]
        category_path = os.path.join(self.root_path, category)
        img_name = os.listdir(category_path)[file_idx]
        img_path = os.path.join(category_path, img_name)

        # Use PIL to open the image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.categories.index(category)

    def _create_index_mapping(self):
        index_mapping = []
        for category_idx, category in enumerate(self.categories):
            category_path = os.path.join(self.root_path, category)
            num_files = len(os.listdir(category_path))
            index_mapping.extend([(category, file_idx) for file_idx in range(num_files)])
        return index_mapping

