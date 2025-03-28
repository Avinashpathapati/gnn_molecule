import os.path as osp
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import OMDBXYZ
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
import logging

target = 0
dim = 32

model_path = os.path.join(os.getcwd(),'model')
if not os.path.exists(model_path):
    os.makedirs('model')

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

print(len(dataset))
dataset = dataset[dataset.data.y > 0]
print(len(dataset))

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
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        nn0 = Sequential(Linear(2, 64), ReLU(), Linear(64, 2*dim*dim))
        nn = Sequential(Linear(2, 64), ReLU(), Linear(64, dim * dim))
        #nn = Sequential(Linear(5, dim * dim))
        self.conv0 = NNConv(dim, 2*dim, nn0, aggr='mean')
        self.gru0 = GRU(2*dim, dim)

        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=1)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        #print('---------------')
        #print(data.x.shape)
        out = F.relu(self.lin0(data.x))
        #print(out.shape)
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        #print(out.shape)
        out = self.set2set(out, data.batch)
        #print(out.shape)
        out = F.relu(self.lin1(out))
        #print(out.shape)
        out = self.lin2(out)
        #print(out.shape)
        #print('-----------------')
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
    
    if osp.isfile('./nnconv.png'):
        os.remove('./nnconv.png') 
    plt.figure()
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Validation')
    plt.plot(epochs, test_loss, label='Test')
    plt.ylabel('Loss [eV]')
    plt.xlabel('epochs')
    plt.title('Validation Test Error Results')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
    plt.savefig('./nnconv.png')
    plt.close()


best_val_error = None
best_test_error = None
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

    if best_val_error is None or val_error <= best_val_error:
        best_test_error = test_error
        best_val_error = val_error
        torch.save(model, os.path.join(model_path, "best_model"))

    

    logging.warning('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f},Best Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error, best_test_error))

plot_results(range(1, 501), train_loss, val_loss, test_loss)
