import torch
import torchtext
import sklearn
import numpy as np
import mpi4py
import gc

from mpi4py import MPI

from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
import keras
import keras.utils
from keras import utils as np_utils
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import keras as k
import operator
import torch.utils.data as data_utils
from torch.utils.data.dataset import random_split





from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)
     





import matplotlib.pyplot as plt
plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()








from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}


import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization






cnn = CNN()


loss_func = nn.CrossEntropyLoss()   


from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   


from torch.autograd import Variable
num_epochs = 10
def train(train_num, num_epochs, cnn, loaders_trains):
  loss_func = nn.CrossEntropyLoss()
  optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
  cnn.train()

  total_step = len(loaders_trains)

  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(loaders_trains):
      batch_x = Variable(images)
      batch_y = Variable(labels)

      output = cnn(batch_x)[0]
      loss = loss_func(output, batch_y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if (i + 1) % 100 == 0:
        print('Train num {}, Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(train_num, 
                                                                              epoch + 1, 
                                                                              num_epochs, 
                                                                              i + 1,
                                                                              total_step,
                                                                              loss.item()))
        pass
      pass
    pass

