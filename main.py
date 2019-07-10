import os
from torchvision import datasets
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import ImageFile
import matplotlib.pyplot as plt
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

### TODO: Write data loaders for training, validation, and test sets
# define training and test data directories
data      = 'data/'
train_dir = os.path.join(data, 'train/')
valid_dir = os.path.join(data, 'valid/')
test_dir  = os.path.join(data, 'test/')

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([transforms.RandomResizedCrop(256),
									 transforms.RandomCrop(224),
									 transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

data_transform_val = transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

# Load data
train_data = datasets.ImageFolder(train_dir, transform=data_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transform_val)
test_data  = datasets.ImageFolder(test_dir, transform=data_transform_val)

## Specify appropriate transforms, and batch_sizes
# define dataloader parameters
batch_size  = 20
num_workers = 0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)

classes = ['melanoma', 'nevus', 'seborrheic_keratosis']

# Visualize some sample data
# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# obtain one batch of training images
dataiter = iter(valid_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    #plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
plt.show()


#########################################
# Model
#########################################
model_conv = models.resnet50(pretrained=True)
num_ftrs = model_conv.fc.in_features
model_conv.fc = torch.nn.Linear(num_ftrs, 3)
print(model_conv.fc)

optimizer = torch.optim.Adam(model_conv.parameters(), lr=1e-6)
criterion = torch.nn.CrossEntropyLoss()

max_epochs = 20
trainings_error = []
validation_error = []
valid_loss_min = np.Inf 
for epoch in range(max_epochs):
    print('epoch:', epoch)
    count_train = 0
    trainings_error_tmp = []
    model_conv.train()
    valid_loss = 0.0
    for data_sample, y in train_loader:
        output = model_conv(data_sample)
        err = criterion(output, y)
        err.backward()
        optimizer.step()
        trainings_error_tmp.append(err.item())
        count_train += 1
        if count_train >= 100:
            count_train = 0
            mean_trainings_error = np.mean(trainings_error_tmp)
            trainings_error.append(mean_trainings_error)
            print('trainings error:', mean_trainings_error)
            break
    with torch.set_grad_enabled(False):
        validation_error_tmp = []
        count_val = 0
        model_conv.eval()
        for batch_idx, (data_sample, y) in enumerate(valid_loader):
            output = model_conv(data_sample)
            err = criterion(output, y)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (err.data - valid_loss))
            validation_error_tmp.append(err.item())
            count_val += 1
            if count_val >= 10:
                count_val = 0
                mean_val_error = np.mean(validation_error_tmp)
                validation_error.append(mean_val_error)
                print('validation error:', mean_val_error)
                break

    ## TODO: save the model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model_conv.state_dict(), 'model_scratch.pt')
        valid_loss_min = valid_loss

plt.plot(trainings_error, label = 'training error')
plt.plot(validation_error, label = 'validation error')
plt.legend()
plt.show()


model_conv.eval()
result_array = []
gt_array = []
for data_sample, y in test_loader:
    output = model_conv(data_sample)
    result = torch.argmax(output)
    result_array.append(result.item())
    gt_array.append(y.item())
correct_results = np.array(result_array)==np.array(gt_array)
sum_correct = np.sum(correct_results)
accuracy = sum_correct/test_loader.__len__()
print(accuracy)