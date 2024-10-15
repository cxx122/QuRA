import torch.nn as nn

# ------------------------------------------------------------------------------
#    Feedforward Neural Network, FFNN
# ------------------------------------------------------------------------------

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNN, self).__init__()
        
        assert hidden_sizes > 0, "need at least one hidden layer"
        
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, hidden_sizes):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def FFNN6(input_size, output_size):
    return FFNN(input_size, 6, output_size)