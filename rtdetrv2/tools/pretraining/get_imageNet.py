import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Specify transformations (optional)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Download the ImageNet dataset for training
imagenet_train = torchvision.datasets.ImageNet(
    root='/data/datasets/ImageNet/',  # Directory to store the dataset
    split='train',  # For training set
    download=True,  # Will download the dataset if not present locally
    transform=transform
)

# Example: Wrap it in a DataLoader
train_loader = DataLoader(imagenet_train, batch_size=4, shuffle=True)
