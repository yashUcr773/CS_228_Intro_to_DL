import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import models
from torchsummary import summary
import os

"""### Create Custom Dataset Class"""

class ImageColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, root='./', transform=None):
        self.root = root
        self.transform = transform
        self.init_dataset()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        image = self.x_train[idx]
        label = self.y_train[idx]
        
        image = Image.open(image)
        label = Image.open(label)

        if self.transform:
            image, label = self.transform(image, label)        

        return (image, label)

    def init_dataset(self):

        from urllib import request
        import zipfile
        import os
        from PIL import Image

        path_to_dataset = 'https://d1u36hdvoy9y69.cloudfront.net/cs-228-intro-to-dl/Project/dataset.zip'
        path_to_zip = f'{self.root}/dataset.zip'
        base_path = f'{self.root}/'
        path_to_bw = f'{base_path}/dataset/bw_images'
        path_to_colored = f'{base_path}/dataset/true_images'


        if not os.path.exists(self.root):
            os.makedirs(self.root)

        request.urlretrieve(path_to_dataset, path_to_zip)
        
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(base_path)

        
        self.x_train = []
        self.y_train = []

        for (dirpath, dirnames, filenames) in os.walk(f'{path_to_bw}'):

            for filename in filenames:
                img_path = dirpath +'/'+ filename
                self.x_train.append(img_path)

                lbl_path = f'{path_to_colored}/'+ filename
                self.y_train.append(lbl_path)

"""### Custom Transforms"""

class CustomRotate(object):
    def __init__(self, rot_low=200, rot_high=360, rot_prob=0.5):
        # Rotation Augmentation
        random_number = torch.randint(rot_low, rot_high, (1,))
        self.angle = random_number.item()
        self.rot_prob = rot_prob
        
    def __call__(self, image, label):
        if torch.rand(1).item() < self.rot_prob:
            image = image.rotate(self.angle)
            label = label.rotate(self.angle)

        return image, label

class CustomVertFlip(object):
    def __init__(self, vert_flip_prob=0.5):
        # Vertical Flip
        self.vert_flip_prob = vert_flip_prob

    def __call__(self, image, label):
        if torch.rand(1).item() < self.vert_flip_prob:
            image = torchvision.transforms.functional.vflip(image)
            label = torchvision.transforms.functional.vflip(label)

        return image, label

class CustomHorFip(object):
    def __init__(self, hor_flip_prob=0.5):
        # Horizontal Flip
        self.hor_flip_prob = hor_flip_prob

    def __call__(self, image, label):
        if torch.rand(1).item() < self.hor_flip_prob:
            image = torchvision.transforms.functional.hflip(image)
            label = torchvision.transforms.functional.hflip(label)

        return image, label
    
class CustomBlur(object):
    def __init__(self, gauss_blur_prob=0.5, gauss_blur_rad = 5):
        # Gaussian Blur
        self.gauss_blur_prob = gauss_blur_prob
        self.gauss_blur_rad = gauss_blur_rad

        
    def __call__(self, image, label):
        if torch.rand(1).item() < self.gauss_blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.gauss_blur_rad))
            label = label.filter(ImageFilter.GaussianBlur(radius=self.gauss_blur_rad))

        return image, label

class CustomToTensor(object):
    def __init__(self):
        pass
        
    def __call__(self, image, label):
        image = torchvision.transforms.functional.to_tensor(image)
        label = torchvision.transforms.functional.to_tensor(label)

        return image, label

class CustomRandomCrop(object):
    def __init__(self, crop_size=(128,128), crop_prob=0.5, n_holes=1):
        self.crop_size = crop_size
        self.crop_prob = crop_prob
        self.n_holes = n_holes

    def __call__(self, image, label):
        _, height, width = image.shape
        mask = torch.ones_like(image)
        
        for _ in range(self.n_holes):
            left = random.randint(0, width - self.crop_size[0])
            top = random.randint(0, height - self.crop_size[1])
            right = left + self.crop_size[0]
            bottom = top + self.crop_size[1]
            
            mask[:, top:bottom, left:right] = 0

        image = image * mask
        return image, label

class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


"""### Create dataloaders"""

transforms = CustomCompose([
    CustomRotate(rot_low=100, rot_high=300, rot_prob=0.5),
    CustomVertFlip(vert_flip_prob=0.5),
    CustomHorFip(hor_flip_prob=0.5),
    CustomBlur(gauss_blur_prob=0.5, gauss_blur_rad=5),
    CustomToTensor(),
    CustomRandomCrop(crop_size=(40,40), crop_prob=1, n_holes=6)
])

image_colorization_dataset = ImageColorizationDataset(root = './dataset', transform=transforms)

train_ratio = 0.6
test_ratio = 0.2
val_ratio = 0.2

train_size = int(len(image_colorization_dataset) * train_ratio)
val_size = int(len(image_colorization_dataset) * val_ratio)
test_size = len(image_colorization_dataset) - train_size - val_size

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(image_colorization_dataset, [train_size, test_size, val_size])

batch_size = 2

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

"""### Use DataLoaders"""

# Iterate over training data loader
print("Training Data:")
for batch_idx, (data, target) in enumerate(train_loader):
    # Print batch index and shape of data and target
    print("Batch Index:", batch_idx)
    print("Data Shape:", data.shape)
    print("Target Shape:", target.shape)
    print()
    break

# Iterate over testing data loader
print("Testing Data:")
for batch_idx, (data, target) in enumerate(test_loader):
    # Print batch index and shape of data and target
    print("Batch Index:", batch_idx)
    print("Data Shape:", data.shape)
    print("Target Shape:", target.shape)
    print()
    break

# Iterate over validation data loader
print("Validation Data:")
for batch_idx, (data, target) in enumerate(val_loader):
    # Print batch index and shape of data and target
    print("Batch Index:", batch_idx)
    print("Data Shape:", data.shape)
    print("Target Shape:", target.shape)
    print()
    break

"""### Visualize Some Samples"""

for batch_idx, (data, target) in enumerate(train_loader):
    # Print batch index and shape of data and target
    print("Batch Index:", batch_idx)
    print("Data Shape:", data.shape)
    print("Target Shape:", target.shape)
    print()
    image = data[0]
    label = target[0]

    plt.figure(1)
    plt.imshow(image.squeeze(0), cmap='gray')
    plt.title('Original')
    plt.plot()
    plt.show()

    plt.figure(1)
    plt.imshow(label.permute(1,2,0))
    plt.title('Labels')
    plt.plot()
    plt.show()
    break

for batch_idx, (data, target) in enumerate(test_loader):
    # Print batch index and shape of data and target
    print("Batch Index:", batch_idx)
    print("Data Shape:", data.shape)
    print("Target Shape:", target.shape)
    print()
    image = data[0]
    label = target[0]

    plt.figure(1)
    plt.imshow(image.squeeze(0), cmap='gray')
    plt.title('Original')
    plt.plot()
    plt.show()

    plt.figure(1)
    plt.imshow(label.permute(1,2,0))
    plt.title('Labels')
    plt.plot()
    plt.show()
    break

for batch_idx, (data, target) in enumerate(val_loader):
    # Print batch index and shape of data and target
    print("Batch Index:", batch_idx)
    print("Data Shape:", data.shape)
    print("Target Shape:", target.shape)
    print()
    image = data[0]
    label = target[0]

    plt.figure(1)
    plt.imshow(image.squeeze(0), cmap='gray')
    plt.title('Original')
    plt.plot()
    plt.show()

    plt.figure(1)
    plt.imshow(label.permute(1,2,0))
    plt.title('Labels')
    plt.plot()
    plt.show()
    break

class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, input):
        encoded = self.encoder(input)
        output = self.decoder(encoded)
        return output

# Define training parameters
learning_rate = 0.001
num_epochs = 300

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create an instance of the colorization model
model = ColorizationModel().to(device)
summary(model, (1, 512, 512))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters:", total_params)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if not os.path.exists('weights'):
    os.makedirs('weights')

if not os.path.exists('images'):
    os.makedirs('images')

# Training loop
for epoch in range(num_epochs):
    print (f'{epoch}/{num_epochs}')
    model.train()
    for i, (images, targets) in enumerate(tqdm(train_loader)):
        # Move images to the device
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), f'weights/colorization_model_encoder_decoder_{epoch}.pth')

    # Load the trained model
    model.eval()
    with torch.no_grad():
        # Generate colorized images
        for i, (images, _) in enumerate(test_loader):
            
            images = images.to(device)
            colorized_images = model(images)
            colorized_images = colorized_images.to('cpu')

            # Create a grid of images
            grid = vutils.make_grid(colorized_images, nrow=4, normalize=True, scale_each=True)

            # Plot the grid of images
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.show()
            plt.savefig(f'images/colorization_model_encoder_decoder_{epoch}.png')
            break