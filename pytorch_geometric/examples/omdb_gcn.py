import os.path as osp
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import OMDBXYZ
from torch_geometric.nn import NNConv, Set2Set, GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

target = 0
dim = 32


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OMDB')
#transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = OMDBXYZ(path).shuffle()

# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.item(), std.item()

# Split datasets.
train_dataset = dataset[:9000]
val_dataset = dataset[9000:10000]
test_dataset = dataset[10000:]

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, 128, cached=True)
        self.conv2 = GCNConv(128, 64, cached=True)
        self.linear = torch.nn.Linear(64, 1)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        #self.reg_params = self.conv1.parameters()
        #self.non_reg_params = self.conv2.parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(edge_index.shape)
        print(x.shape)
        print(data.num_features)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        return x.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=25,
                                                       min_lr=1e-6)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


def plot_results(epochs, train_loss, val_loss, test_loss):
    
    if osp.isfile('./gcnconv.png'):
        os.remove('./gcnconv.png') 
    plt.figure()
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Validation')
    plt.plot(epochs, test_loss, label='Test')
    plt.ylabel('Loss [eV]')
    plt.xlabel('epochs')
    plt.title('Validation Test Error Results')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
    plt.savefig('./gcnconv.png')
    plt.close()


best_val_error = None
train_loss = []
val_loss = []
test_loss = []
for epoch in range(1, 501):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)
    train_loss.append(loss)
    val_loss.append(val_error)
    test_error = test(test_loader)
    test_loss.append(test_error)

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

plot_results(range(1, 501), train_loss, val_loss, test_loss)