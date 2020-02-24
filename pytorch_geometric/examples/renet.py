import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ICEWS18, GDELT  # noqa
from torch_geometric.data import DataLoader
from torch_geometric.nn.models.re_net import RENet

seq_len = 10

# Load the dataset and precompute history objects.
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ICEWS18')
train_dataset = ICEWS18(path, pre_transform=RENet.pre_transform(seq_len))
test_dataset = ICEWS18(path, split='test')

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'GDELT')
# train_dataset = GDELT(path, pre_transform=RENet.pre_transform(seq_len))
# test_dataset = ICEWS18(path, split='test')

# Create dataloader for training and test dataset.
train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    follow_batch=['h_sub', 'h_obj'],
    num_workers=6)
test_loader = DataLoader(
    test_dataset,
    batch_size=1024,
    shuffle=False,
    follow_batch=['h_sub', 'h_obj'],
    num_workers=6)

# Initialize model and optimizer.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RENet(
    train_dataset.num_nodes,
    train_dataset.num_rels,
    hidden_channels=200,
    seq_len=seq_len,
    dropout=0.5,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.00001)


def train():
    model.train()

    # Train model via multi-class classification against the corresponding
    # object and subject entities.
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        log_prob_obj, log_prob_sub = model(data)
        loss_obj = F.nll_loss(log_prob_obj, data.obj)
        loss_sub = F.nll_loss(log_prob_sub, data.sub)
        loss = loss_obj + loss_sub
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    # Compute Mean Reciprocal Rank (MRR) and Hits@1/3/10.
    result = torch.tensor([0, 0, 0, 0], dtype=torch.float)
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            log_prob_obj, log_prob_sub = model(data)
        result += model.test(log_prob_obj, data.obj) * data.obj.size(0)
        result += model.test(log_prob_sub, data.sub) * data.sub.size(0)
    result = result / (2 * len(loader.dataset))
    return result.tolist()


for epoch in range(1, 21):
    train()
    results = test(test_loader)
    print('Epoch: {:02d}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, '
          'Hits@10: {:.4f}'.format(epoch, *results))
