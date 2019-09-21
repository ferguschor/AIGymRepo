import torch.nn as nn
import torch.nn.functional as F


class SimpleNet (nn.Module):

    def __init__(self, num_inputs, H, num_outputs):
        super(SimpleNet, self).__init__()
        # self.num_hidden = (num_inputs + num_outputs) / 2
        self.num_hidden = H
        self.linear1 = nn.Linear(num_inputs, self.num_hidden)
        self.linear2 = nn.Linear(self.num_hidden, num_outputs)


    # forward function does not include softmax. Softmax is included in criterion.
    # Softmax needs to be implemented during predict.
    def forward(self, x):
        hidden_pred = F.relu(self.linear1(x))
        y_pred = self.linear2(hidden_pred)
        return y_pred