import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from IrisDataLoader import Dataset
from torch.utils import data
import time


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class SimpleNet (nn.Module):

    def __init__(self, num_inputs, H, num_outputs):
        super(SimpleNet, self).__init__()
        # self.num_hidden = (num_inputs + num_outputs) / 2
        self.num_hidden = H
        self.linear1 = nn.Linear(num_inputs, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, num_outputs)

    def forward(self, x):
        hidden_pred = F.relu(self.linear1(x))
        y_pred = self.linear2(hidden_pred)
        return y_pred


train_ratio = 0.7

# Load iris data into training dataset and labels
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=42)


print(x_train)
print(y_train)

x = torch.Tensor(x)
y = torch.LongTensor(y)
x_train = torch.Tensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.Tensor(x_test)
y_test = torch.LongTensor(y_test)


train_dataset = Dataset(x_train, y_train)
train_dataloader = data.DataLoader(train_dataset, batch_size=10)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = int(len(x) * train_ratio), 4, 20, 3

print ("X:\n", x)
print ("Y:\n", y)

model = SimpleNet(D_in, H, D_out)

rand = random.choices(range(N), k=10)
print(rand)


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# criterion = torch.nn.MSELoss(reduction="sum")
crit = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.90)

list_train_loss = []
list_test_loss = []

start = time.time()

for t in range(2500):

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)
    # Computer and print loss
    loss = crit(y_pred, y_train)
    print (t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 9 == 0:
        y_test_pred = model(x_test)
        test_loss = crit(y_test_pred, y_test)
        list_test_loss.append([t, test_loss.item()])
        print("Test Error: {}".format(test_loss))

    list_train_loss.append([t, loss.item()])

end = time.time()

train_err = np.array(list_train_loss)
print(train_err)
test_err = np.array(list_test_loss)
print(test_err)

for i in model.parameters():
    print (i.size())

print("Time: %f" % (end-start))

# plt.plot(train_err[:,0], train_err[:,1], label="train")
# plt.plot(test_err[:,0], test_err[:,1], label="test")
# plt.legend(loc='upper left')
# plt.show()


# Find the model accuracy
sm = nn.Softmax(dim=1)
print (sm(y_test_pred))
values, indices = torch.max(sm(y_test_pred), 1)
match = indices == y_test
print (torch.mean(match.type(torch.DoubleTensor)))

# Train basic logistic regression and compare accuracy
ln = LogisticRegression()
ln.fit(x_train, y_train)
ln_y_test = torch.from_numpy(ln.predict(x_test))
print (torch.mean((y_test==ln_y_test).type(torch.DoubleTensor)))
