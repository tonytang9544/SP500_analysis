import torch

class MLP_predictor(torch.nn.Module):
    def __init__(self, number_stocks, hidden_dim=256, num_hidden_layers=3, activation=torch.nn.Tanh(), dropout_p=0.1):
        super().__init__()
        self.projection = torch.nn.Linear(number_stocks, hidden_dim)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.output = torch.nn.Linear(hidden_dim, number_stocks)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_p)
    
    def forward(self, state):
        '''
        state is vector of one-day price change
        '''
        x = self.projection(state)
        for each_layer in self.layers:
            x = self.dropout(self.activation(each_layer(x)))
        return self.output(x) + state