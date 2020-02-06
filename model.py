import torch.nn as nn
import torch.nn.functional as F


class DAR(nn.Module):
    '''
    A simple, general purpose, fully connected network
    '''

    def __init__(self, ar, num_layers=1, d_hidden=None):
        # Perform initialization of the pytorch superclass
        super().__init__()
        # Define network layer dimensions
        d_in, d_out = [ar, 1]
        self.ar = ar
        self.num_layers = num_layers
        if d_hidden is None and num_layers > 1:
            d_hidden = d_in
        if self.num_layers == 1:
            self.layer_1 = nn.Linear(d_in, d_out, bias=True)
        else:
            self.layer_1 = nn.Linear(d_in, d_hidden, bias=True)
            self.mid_layers = []
            for i in range(self.num_layers - 2):
                self.mid_layers.append(nn.Linear(d_hidden, d_hidden, bias=True))
            self.layer_out = nn.Linear(d_hidden, d_out, bias=True)

    def forward(self, x):
        '''
        This method defines the network layering and activation functions
        '''
        activation = F.relu
        x = self.layer_1(x)
        if self.num_layers > 1:
            x = activation(x)
            for layer in self.mid_layers:
                x = layer(x)
                x = activation(x)
            x = self.layer_out(x)
        return x


def main():
    pass


if __name__ == "__main__":
    main()
