# Coriander vs Parsley Classifier

Cooking for oneself can be both satisfying and challenging—especially when you can’t tell the difference between **coriander** and **parsley**! As someone who lives alone, I often found myself facing this exact problem. To solve it, I decided to leverage **machine learning** to build a classifier that can **distinguish between coriander and parsley** accurately.


---

## Additional Information:
This notebook is part of a larger project where the classifier is deployed as a **FastAPI** API, and a **front-end template** allows users to interact with the API. Users can upload images of herbs to get predictions on whether the image is of coriander or parsley.

- **API Backend**: [API](https://github.com/echarif/coriander_vs_parsley_api)
- **Front-End Template**: [[Template](https://github.com/echarif/coriander_vs_parsley_template)]

---
## Development:

### **Importing Libraries and Configurations**  
Sets up the environment for the project by importing all necessary libraries. It configures the plotting settings for better visual output, enables high-resolution figures, and imports libraries like `torch`, `torchvision`, `PIL`, and others needed for image processing, model building, and visualization. It also configures the system to help debug GPU issues with CUDA by setting an environment variable.

```python
%matplotlib inline  
%config InlineBackend.figure_format = 'retina'  
import torchvision  
import gc  
import time  
from torchvision import transforms, models, datasets  
import torch  
import numpy as np  
import matplotlib.pyplot as plt  
import torch.optim as optim  
import torch.nn as nn  
from collections import OrderedDict  
from PIL import Image  
import seaborn as sns  
import helper  
import numpy as np  
import pandas as pd  
import json  
import os  
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  
```
### **Mounting Google Drive**  
Mounts Google Drive to the Colab environment, enabling access to files stored in the drive.

```python
# Load the Drive helper and mount
from google.colab import drive
drive.mount('/content/drive')
```  
```
Mounted at /content/drive
```

### **Data Loading and Transformation**  
This cell defines data transformations for both training and testing datasets, loads the images from Google Drive, and applies the transformations. It then prepares the datasets and loads them into DataLoader objects for batching and shuffling.

```python
train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
train_datasets = datasets.ImageFolder('/content/drive/My Drive/Coriander_vs_Parsley/train', transform=train_transforms)
test_datasets = datasets.ImageFolder('/content/drive/My Drive/Coriander_vs_Parsley/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)

print("train size: ", len(trainloader.dataset))
print("test size: ", len(testloader.dataset))
```

```
train size:  193
test size:  51
```
### **Image Display Function**  
defines a custom function `imshow()` to display images from a tensor, applying necessary normalization for visualization. It then displays an image from the training dataset and prints its corresponding label.

```python
def imshow(image, ax=None, title=None, normalize=True):
  """Imshow for Tensor."""
  if ax is None:
      fig, ax = plt.subplots()
  image = image.numpy().transpose((1, 2, 0))

  if normalize:
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      image = std * image + mean
      image = np.clip(image, 0, 1)

  ax.imshow(image)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(axis='both', length=0)
  ax.set_xticklabels('')
  ax.set_yticklabels('')

  return ax

data_iter = iter(trainloader)
images, labels = next(data_iter)
imshow(images[0])
print(labels[0])
```

```
tensor(0)
```

The output shows the label `0`, indicating the first image belongs to class `0`.

### **Load Pretrained Model**  
This cell loads the pre-trained DenseNet-201 model and modifies it by enabling gradient computation for all parameters. The model is downloaded, and a warning is issued regarding the deprecated use of `pretrained=True`.

```python
# we will use a pretrained model and we are going to change only the last layer
model = models.densenet201(pretrained=True)
for param in model.parameters():
  param.requires_grad = True
```

```
/usr/local/lib/python3.10.dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.10.dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=DenseNet201_Weights.IMAGENET1K_V1`. You can also use `weights=DenseNet201_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/densenet201-c1103571.pth" to /root/.cache/torch/hub/checkpoints/densenet201-c1103571.pth
100%|██████████| 77.4M/77.4M [00:00<00:00, 218MB/s]
```

### **Modify the Classifier Layer**  
Replace the original classifier of the DenseNet-201 model with a new custom classifier. The new classifier consists of two fully connected layers with a ReLU activation in between, followed by a LogSoftmax layer for multi-class classification.

```python
classifier = nn.Sequential(nn.Linear(1920, 256),
                          nn.ReLU(),
                          nn.Linear(256, 2),
                          nn.LogSoftmax(dim=1))
model.classifier = classifier
```

### **Set Device and Initialize Training Components**  
This cell checks if CUDA (GPU support) is available, and then moves the model to the appropriate device (either GPU or CPU). It also sets up the loss function (`NLLLoss`) and the optimizer (`Adam`), configuring them to only update the parameters of the custom classifier. Finally, it initializes the `test_loss_min` variable and sets the file name for saving the model.

```python
if torch.cuda.is_available():
  model.to('cuda')
  device = 'cuda'
else:
    model.to('cpu')
    device = 'cpu'
print(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.00001, weight_decay=0)
test_loss_min = 99  # just a big number I could do np.Inf
save_file = 'mymodel.pth'
```

```
cuda
```  
The model is successfully moved to the GPU (if available), and the device is set to `cuda`.


### **Cell 8: Model Training Loop with Evaluation**  
In this cell, the model is trained for 200 epochs. The training and evaluation (test) phases are performed within each epoch. During the training phase, the model computes predictions, calculates the loss, performs backpropagation, and updates the weights. After every 10 epochs, the test loss and accuracy are evaluated to check the model's performance on the test set. If the test loss improves, the model is saved to a file (`mymodel.pth`). The time taken for each epoch is also printed for tracking the training process.

```python
epochs = 200
train_losses = []
test_losses = []
print_every = 10
running_loss = 0
for epoch in range(epochs):
    time0 = time.time()
    model.train()
    for inputs, labels in trainloader:
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        prediction = model.forward(inputs)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    else:
        train_losses.append(running_loss / len(trainloader))
        running_loss = 0
        if ((epoch % print_every) == 0):
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                total_loss = test_loss / len(testloader)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss / (len(trainloader) * print_every):.3f}.. "
                      f"test loss: {test_loss / len(testloader):.3f}.. "
                      f"test accuracy: {accuracy / len(testloader):.3f}")
                time_total = time.time() - time0
                print("time for this epoch: ", end="")
                print(time_total)

                test_losses.append(total_loss)
                if total_loss <= test_loss_min:
                    print(f'test loss decreased ({test_loss_min:.6f} --> {total_loss:.6f}).  Saving model ...')
                    torch.save(model.state_dict(), save_file)
                    test_loss_min = total_loss
                running_loss = 0
```

```
Epoch 1/200.. Train loss: 0.000.. test loss: 0.699.. test accuracy: 0.431
time for this epoch: 82.6384539604187
test loss decreased (99.000000 --> 0.699303).  Saving model ...
Epoch 11/200.. Train loss: 0.000.. test loss: 0.675.. test accuracy: 0.569
time for this epoch: 4.93536376953125
test loss decreased (0.699303 --> 0.675338).  Saving model ...
Epoch 21/200.. Train loss: 0.000.. test loss: 0.674.. test accuracy: 0.569
time for this epoch: 5.192459583282471
test loss decreased (0.675338 --> 0.674152).  Saving model ...
Epoch 31/200.. Train loss: 0.000.. test loss: 0.679.. test accuracy: 0.569
time for this epoch: 4.868926048278809
Epoch 41/200.. Train loss: 0.000.. test loss: 0.675.. test accuracy: 0.569
time for this epoch: 5.283760070800781
Epoch 51/200.. Train loss: 0.000.. test loss: 0.661.. test accuracy: 0.569
time for this epoch: 4.880338907241821
test loss decreased (0.674152 --> 0.660619).  Saving model ...
Epoch 61/200.. Train loss: 0.000.. test loss: 0.656.. test accuracy: 0.588
time for this epoch: 4.974986791610718
test loss decreased (0.660619 --> 0.655996).  Saving model ...
Epoch 71/200.. Train loss: 0.000.. test loss: 0.648.. test accuracy: 0.588
time for this epoch: 5.145587205886841
test loss decreased (0.655996 --> 0.648160).  Saving model ...
Epoch 81/200.. Train loss: 0.000.. test loss: 0.650.. test accuracy: 0.588
time for this epoch: 4.875314712524414
Epoch 91/200.. Train loss: 0.000.. test loss: 0.641.. test accuracy: 0.745
time for this epoch: 5.285011053085327
test loss decreased (0.648160 --> 0.641278).  Saving model ...
Epoch 101/200.. Train loss: 0.000.. test loss: 0.645.. test accuracy: 0.569
time for this epoch: 4.876373767852783
Epoch 111/200.. Train loss: 0.000.. test loss: 0.640.. test accuracy: 0.608
time for this epoch: 4.962859392166138
test loss decreased (0.641278 --> 0.639720).  Saving model ...
Epoch 121/200.. Train loss: 0.000.. test loss: 0.629.. test accuracy: 0.706
time for this epoch: 5.130331039428711
test loss decreased (0.639720 --> 0.628545).  Saving model ...
Epoch 131/200.. Train loss: 0.000.. test loss: 0.636.. test accuracy: 0.569
time for this epoch: 4.892789125442505
Epoch 141/200.. Train loss: 0.000.. test loss: 0.654.. test accuracy: 0.569
time for this epoch: 5.295125722885132
Epoch 151/200.. Train loss: 0.000.. test loss: 0.620.. test accuracy: 0.647
time for this epoch: 4.869210243225098
test loss decreased (0.628545 --> 0.620273).  Saving model ...
Epoch 161/200.. Train loss: 0.000.. test loss: 0.642.. test accuracy: 0.569
time for this epoch: 5.007403135299683
Epoch 171/200.. Train loss: 0.000.. test loss: 0.634.. test accuracy: 0.588
time for this epoch: 5.169550180435181
Epoch 181/200.. Train loss: 0.000.. test loss: 0.628.. test accuracy: 0.647
time for this epoch: 4.876027345657349
Epoch 191/200.. Train loss: 0.000.. test loss: 0.620.. test accuracy: 0.647
time for this epoch: 5.131802797317505
test loss decreased (0.620273 --> 0.619859).  Saving model ...

```

- The training process runs for 200 epochs, and every 10 epochs the test loss and accuracy are calculated.
- The model's performance is tracked, and if the test loss improves (is lower), the model is saved to the file `mymodel.pth`.
- The training loss and test metrics (loss and accuracy) are displayed for every 10th epoch.


### **Plotting Training Loss Over Epochs**

This cell generates a plot showing the progression of training loss throughout the training process. It uses Matplotlib to visualize how the model's performance improves (or fluctuates) as it learns from the training data.

```python
plt.plot(train_losses)
# plt.plot([k for k in range(0, epochs, print_every)], test_losses)
plt.show()
```
### **Saving the Model Weights**

This cell saves the trained model weights to a file on Google Drive. The model's state dictionary, which contains all the learned parameters (weights and biases), is saved in the specified path. This allows for the model to be reloaded later without retraining.

```python
torch.save(model.state_dict(), '/content/drive/My Drive/Coriander_vs_Parsley/coriander_vs_parsley_model_weights.pth')
```


### **Making Predictions with the Saved Model**

To make new predictions using the trained model, you can use the file `coriander_vs_parsley_new_prediction.ipynb`. Simply load the model, apply the necessary transformations to your input data, and then pass it through the model for inference.
