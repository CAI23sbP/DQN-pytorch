import torch.nn as nn

class QValue(nn.Module):
    def __init__(self, configs):
        super(QValue, self).__init__()
        self.q_value = self.mlp(config=configs)
        print(self.q_value)

    def mlp(self, config, last_active=True) -> nn.Sequential:
        layers = []
        mlp_dims = [config.Env.observation_space] + config.Network.mlp_dims
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2 or last_active:
                layers.append(config.Network.activation)
        net = nn.Sequential(*layers)
        return net

    def forward(self,X):
        Y = self.q_value(X)
        return Y
