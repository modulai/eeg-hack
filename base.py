import hand_rnn as hand_rnn
import pandas as pd
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# TODO: fix gpu code

#torch.cuda.set_device(0)
device = torch.device("cpu")
#torch.backends.cudnn.benchmark = True


batch_size = 1
epochs = 100

# Learning rate
lr = 0.001
# Betas
betas = (0.9, 0.999)
# Regularisation
weight_decay = 0.0001

(input_tensors, events_tensors)\
   = pickle.load(open(f"data/tensors.pkl", "rb"))

# split into two files

ds = data.TensorDataset(input_tensors, events_tensors)
dataloader_ = data.DataLoader(ds,
                              batch_size=batch_size,
                              shuffle=True)

hand_rnn = hand_rnn.HandRNN()

optimizer = optim.Adam(hand_rnn.parameters(), amsgrad=True,
                       lr=lr,
                       betas=betas,
                       weight_decay=weight_decay)

criterion = nn.MSELoss()

for e in range(epochs):
# Transfer the tensors to the GPU
    for i_batch, ts_ in enumerate(dataloader_):
        optimizer.zero_grad()
        y = hand_rnn(ts_[0], device=device)
        loss = criterion(y, ts_[1])
        print("Epoch: " + str(e) + " Batch: " + str(i_batch)
              + " Loss: " + str(loss.item()))
        loss.backward()
        optimizer.step()
