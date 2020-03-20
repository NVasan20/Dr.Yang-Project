import torch
import numpy as np
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


#Data Preparation

dataset = MNIST(root = 'data/', download = True, transform = ToTensor())

#Randomly select 20% of images from dataset

def splitIndices (n, valPct):
    nVal = int(valPct*n)
    indices = np.random.permutation(n)
    return indices[nVal:], indices[:nVal]

#Use Split Indicies to select training and value indicies

train_indices = splitIndices(len(dataset),valPct=0.2)
val_indices = train_indices
print(len(train_indices), len(val_indices))
print("sample Value Indicies", val_indices[:20])

batchSize = 100

#Create Pytorch Dataloaders

trainSampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset, batchSize, sampler = trainSampler)
#validation Sampler
validSampler = SubsetRandomSampler(val_indices)
valid_dl = DataLoader(dataset, batchSize, sampler = validSampler)

#Construct Neural Network with hidden layer
class MnistModule (nn.Module):
    def __init__(self, inSize, hiddenSize, outSize):
        super().__init__()
        #HIdden Layer
        self.linear1 = nn.Linear(inSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outSize)
    def forward(self, xb):
        #flatten Image tensors
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        return out
    
input_size = 784
hiddenSize =32
num_classes = 10
outSize = num_classes
model = MnistModule(input_size, hiddenSize, outSize)

for t in model.parameters():
    print(t.shape)
    
for images, labels in train_dl:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss = ',   loss.item)
    break

#Helper function to identidy GPU, and implement it if available
    
def getDevice():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else: 
        return torch.device('cpu')
    
device = getDevice

#Helper FUnction to move data to chosen device GPU or CPU

def toDevice (data, device):
    if isinstance(data, (list,tuple)):
        return [toDevice(x, device) for x in data]
    return data.to(device, non_blocking = True)

class deviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield toDevice(b, self.device)
            
    def __len__(self):
        return len(self.dl)
    
#Upload dataloaders on device
        
train_dl = deviceDataLoader(train_dl, device)
valid_dl = deviceDataLoader(valid_dl, device)

#Apply Model using GPU and define loss 

def loss_batch(model, loss_func, xb, yb, opt = None, metric = None):
    preds = model(xb)
    loss = loss_func(preds, yb)
    
    #optional- compute gradients
    if opt is not None :
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)
    return loss.item, metric_result, len(xb)

def evaluate(model, lossFn, valid_dl, metric =None):
    with torch.no_grad():
        results = [loss_batch(model, lossFn, xb, yb, metric = metric) 
                   for xb,yb in valid_dl]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums))/total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metric,nums)) / total
    return avg_loss, avg_metric, total

#create fit function with customizeable learning rate, print validation loss and accuracy every epock

def fit(epochs, lossFn, lr, model, train_dl, valid_dl, metric = None, opt_fn = None ):
    losses, metrics = [], []
    
    if opt_fn is not None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr = lr)
    
    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss,_,_ = loss_batch(model, lossFn, xb, yb, opt)
            
        #Evaluate
        result = evaluate(model, lossFn, valid_dl, metric)
        val_loss, total, val_metric = result
        
        losses.append(val_loss)
        metrics.append(val_metric)
        
        if metric is None:
            print('Epoch{}/{}, Loss{:.4f}'.format(epoch+1, epochs, val_loss))
            
        else:
            print('Epoch{}/{}, Loss{:.4f}, {}, {:.4f}'.format(epoch + 1, epochs,
                                                              val_loss, metric.__name__, val_metric ))
        return losses, metrics
    
#accuracy used as metric in fit function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == labels).item / len(preds)

model = MnistModule(input_size, hiddenSize, outSize)
toDevice(model, device)
        
val_loss, total, val_acc = evaluate(model, F.cross_entropy, valid_dl, metric=accuracy)
print("Accuracy: {:.4f}, Loss: {:.4f}".format(val_acc, val_loss))

#Train nn for 5 epochs 94% accuracy learning rate 0.5

losses1, metrics1 = fit(5, F.cross_entropy, 0.5, model, train_dl, valid_dl, 
                        accuracy)  

#train 5 more epochs, learning rate 0.1

losses2, metrics2 = fit(5, F.cross_entropy, 0.1, model, train_dl, valid_dl, 
                        accuracy)

accuracies = [val_acc] + metrics1 +metrics2
plt.plot(accuracies, "-x")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("accuracy v. epochs")


                     
        
        

        
    
    
