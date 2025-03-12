import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import Dataset, ConcatDataset, DataLoader

<<<<<<< HEAD

=======
# Define the image transformation and augmentation pipelines
>>>>>>> ad813dbc454771d089fffab550144744d39ed56c
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# Define a custom Dataset class that applies augmentations
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        sample = self.original_dataset[index]
        img, label = sample['img'], sample['class']

<<<<<<< HEAD
=======
        # Apply the transformation if provided
>>>>>>> ad813dbc454771d089fffab550144744d39ed56c
        if self.transform:
            img = self.transform(img)

        return {'img': img, 'class': torch.tensor(label, dtype=torch.long)}

<<<<<<< HEAD

=======
# Inside data_augmentation.py
>>>>>>> ad813dbc454771d089fffab550144744d39ed56c
def load_data(data_augmentation=False):
    """
    Loads the original dataset and applies transformations, optionally combining with augmented data.
    """
    dataset = load_dataset("cvdl/oxford-pets")
<<<<<<< HEAD
    original_train_set = dataset['train']
    original_valid_set = dataset['test'] 
=======
    original_train_set = dataset['train']  # assuming 'train' split
    original_valid_set = dataset['test']  # assuming 'test' split
>>>>>>> ad813dbc454771d089fffab550144744d39ed56c

    if data_augmentation:
        augmented_train_set = AugmentedDataset(original_train_set, transform=augmentation_transforms)
        combined_train_set = ConcatDataset([original_train_set, augmented_train_set])
    else:
        combined_train_set = original_train_set

    # Create data loaders for both training and validation sets
    data_loader_train = DataLoader(combined_train_set, batch_size=32, shuffle=True)
    data_loader_valid = DataLoader(original_valid_set, batch_size=32)

    return data_loader_train, data_loader_valid

def load_and_combine_data():
    """
    Loads the dataset, applies augmentations, and combines the original dataset with augmented data.
    """
    # Load the original training dataset
    dataset = load_dataset("cvdl/oxford-pets")
    dataset = dataset.select_columns(["img", "class"])
<<<<<<< HEAD
    original_train_set = dataset["train"] 
    
    print(f"Original training set size: {len(original_train_set)} images")

    # Create augmented dataset
    augmented_train_set = AugmentedDataset(original_train_set, transform=augmentation_transforms)
    print(f"Augmented training set size: {len(augmented_train_set)} images")
    # Combine the original dataset with the augmented dataset
    combined_train_set = ConcatDataset([original_train_set, augmented_train_set])
    print(f"Combined training set size: {len(combined_train_set)} images (Original + Augmented)")
=======
    original_train_set = dataset["train"]  # assuming 'train' split
    
    # Create augmented dataset
    augmented_train_set = AugmentedDataset(original_train_set, transform=augmentation_transforms)
    
    # Combine the original dataset with the augmented dataset
    combined_train_set = ConcatDataset([original_train_set, augmented_train_set])

>>>>>>> ad813dbc454771d089fffab550144744d39ed56c
    return combined_train_set

def save_combined_dataset():
    """
    Combines the dataset and saves it as a .pth file.
    """
    combined_train_set = load_and_combine_data()
    torch.save(combined_train_set, "combined_train_set.pth")
    print("Combined dataset saved as 'combined_train_set.pth'.")

if __name__ == "__main__":
    save_combined_dataset()
