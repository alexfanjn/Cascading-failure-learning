import numpy as np
import torch
from torch.nn import Module, Linear
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from utils import load_data, load_topology, load_nm2_changed, load_nm3to6
import torch.utils.data as Data
import random


class FCNet(Module):
    def __init__(self, output_dim):
        super(FCNet, self).__init__()
        self.fc1 = Linear(353, 100)
        self.fc2 = Linear(100, 50)
        self.fc3 = Linear(50, output_dim)


    def forward(self, x, non_zero_index):
        x = x[:, non_zero_index[0], non_zero_index[1]]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':

    current_seed_add = 2022

    setup_seed(current_seed_add)


    # you can customize you own classes based on your practical scenarios.
    split_list = [100, 1000]

    # load power information including ori n-2 data, modified n-2 data, and n-3 data
    before_array, after_array, y = load_data(split_list)
    before_array_test1, after_array_test1, y_test1 = load_nm2_changed(split_list)
    before_array_test2, after_array_test2, y_test2 = load_nm3to6(3, split_list)
    before_array_test4, after_array_test4, y_test4 = load_nm3to6(4, split_list)
    before_array_test5, after_array_test5, y_test5 = load_nm3to6(5, split_list)
    before_array_test6, after_array_test6, y_test6 = load_nm3to6(6, split_list)

    # load uiuc 150 adj
    adj = load_topology()

    # add self loop
    adj = adj + np.eye(adj.shape[0])

    # obtain edge index, non_zero_index will include the (i, j) index of each non zero value in adj (with (i, i))
    triangle_adj = np.triu(adj)
    non_zero_index = np.nonzero(triangle_adj)


    # generate id index for split train and test set
    idx = np.arange(before_array.shape[0])

    # split train and test set following the original label distribution
    idx_train, idx_test,_ , _ = train_test_split(idx, y, test_size=0.3, stratify=y)
    # print(idx_train)
    # print(idx_test)

    # mlp net
    net = FCNet(np.max(y)+1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # define the weighted loss
    weights = np.bincount(y, minlength=np.max(y) + 1)
    weights = 1 / weights


    # transform all data to tensor form for pytorch training
    weights = torch.tensor(weights, dtype=torch.float)
    criterion = torch.nn.NLLLoss(weight=weights)



    # one_hot_y = F.one_hot(torch.tensor(y), num_classes = np.max(y) + 1)
    y = torch.tensor(y)
    y_test1 = torch.tensor(y_test1)
    y_test2 = torch.tensor(y_test2)
    y_test4 = torch.tensor(y_test4)
    y_test5 = torch.tensor(y_test5)
    y_test6 = torch.tensor(y_test6)


    # constract the delta matrix
    X = after_array - before_array
    X_test1 = after_array_test1 - before_array_test1
    X_test2 = after_array_test2 - before_array_test2
    X_test4 = after_array_test4 - before_array_test4
    X_test5 = after_array_test5 - before_array_test5
    X_test6 = after_array_test6 - before_array_test6

    X = torch.tensor(X, dtype=torch.float32)
    X_test1 = torch.tensor(X_test1, dtype=torch.float32)
    X_test2 = torch.tensor(X_test2, dtype=torch.float32)
    X_test4 = torch.tensor(X_test4, dtype=torch.float32)
    X_test5 = torch.tensor(X_test5, dtype=torch.float32)
    X_test6 = torch.tensor(X_test6, dtype=torch.float32)

    # buide torch dataset with batch
    torch_dataset = Data.TensorDataset(X[idx_train], y[idx_train])
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=512,
        shuffle=True,
        drop_last=False
    )

    print('begin training')
    epoch = 0
    max_epoch = 2000
    loss_list = []
    acc_list = []
    test_list = []
    while epoch < max_epoch:
        for step, (batch_x, batch_y) in enumerate(loader):
            out = net(batch_x, non_zero_index)

            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == max_epoch - 1:
            out = net(X, non_zero_index)
            pred = torch.argmax(out, dim=1)
            a = criterion(out, y)
            correct = (pred[idx_train] == y[idx_train]).sum().float()
            acc = correct/len(idx_train)
            test_correct = (pred[idx_test] == y[idx_test]).sum().float()
            test_acc = test_correct/len(idx_test)
            print('Epoch:', epoch, ', loss: ', a.data.item(), ', train_acc = ', acc.numpy(), ', test_acc = ', test_acc.numpy())

        epoch += 1






    print('\nbegin ori training set testing')
    out = net(X, non_zero_index)
    pred = torch.argmax(out, dim=1)
    correct = (pred[idx_train] == y[idx_train]).sum().float()
    acc = correct / len(idx_train)
    print('final ori training acc: ', acc.numpy())


    print('\nbegin ori testing')
    out = net(X, non_zero_index)
    pred = torch.argmax(out, dim=1)
    correct = (pred[idx_test] == y[idx_test]).sum().float()
    acc = correct / len(idx_test)
    print('final ori test acc: ', acc.numpy())



    print('\nbegin nm2_changed testing')
    out = net(X_test1, non_zero_index)
    pred = torch.argmax(out, dim=1)
    correct = (pred == y_test1).sum().float()
    acc = correct / len(pred)
    print('final nm2_changed acc: ', acc.numpy())



    print('\nbegin nm3 testing')
    out = net(X_test2, non_zero_index)
    pred = torch.argmax(out, dim=1)
    correct = (pred == y_test2).sum().float()
    acc = correct / len(pred)
    print('final nm3 acc: ', acc.numpy())


    print('\nbegin nm4 testing')
    out = net(X_test4, non_zero_index)
    pred = torch.argmax(out, dim=1)
    correct = (pred == y_test4).sum().float()
    acc = correct / len(pred)
    print('final nm4 acc: ', acc.numpy())



    print('\nbegin nm5 testing')
    out = net(X_test5, non_zero_index)
    pred = torch.argmax(out, dim=1)
    correct = (pred == y_test5).sum().float()
    acc = correct / len(pred)
    print('final nm5 acc: ', acc.numpy())



    print('\nbegin nm6 testing')
    out = net(X_test6, non_zero_index)
    pred = torch.argmax(out, dim=1)
    correct = (pred == y_test6).sum().float()
    acc = correct / len(pred)
    print('final nm6 acc: ', acc.numpy())