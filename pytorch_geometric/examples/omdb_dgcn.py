import os.path as osp
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin

import torch_geometric.transforms as T
from torch_geometric.datasets import OMDBXYZ
from torch_geometric.nn import NNConv, DynamicEdgeConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

from pointnet2_classification import MLP
import logging

target = 0
dim = 32


class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OMDB')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False), 
    T.RandomTranslate(0.01), T.RandomRotate(15, axis=0), T.RandomRotate(15, axis=1), T.RandomRotate(15, axis=2)])
dataset = OMDBXYZ(path,transform=transform).shuffle()

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
    def __init__(self, k=30, aggr='max'):
        super(Net, self).__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 4, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Lin(128, 1))

    def forward(self, data):
        x, pos, batch = data.x, data.pos, torch.zeros(1, dtype=torch.long)
        x0 = torch.cat([x, pos], dim=-1)
        print(x0.shape)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        return out.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.8, patience=25,
                                                       min_lr=1e-6)


def train(epoch):
    model.train()
    loss_all = 0

    for data_batch in train_loader:
        #to extract data object from batch
        data_batch = data_batch.to(device)
        data_list = data_batch.to_data_list()
        optimizer.zero_grad()
        loss_batch = 0
        for data in data_list:
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            loss_batch += loss.item()
        loss_all += loss_batch
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data_batch in loader:
        data_batch = data_batch.to(device)
        data_list = data_batch.to_data_list()
        for data in data_list:
            error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    
    return error / len(loader.dataset)


def plot_results(epochs, train_loss, val_loss, test_loss):
    
    if osp.isfile('./dgcn.png'):
        os.remove('./dgcn.png') 
    plt.figure()
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Validation')
    plt.plot(epochs, test_loss, label='Test')
    plt.ylabel('Loss [eV]')
    plt.xlabel('epochs')
    plt.title('Validation Test Error Results')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
    plt.savefig('./dgcn.png')
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

    logging.warning('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

plot_results(range(1, 501), train_loss, val_loss, test_loss)
