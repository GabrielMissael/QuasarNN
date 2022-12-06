from torch import nn, optim

class FCVanilla(nn.Module):
    def __init__(self, layers_dims):
        super().__init__()
        self.type = 'FCVanilla'
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential()

        for i in range(len(layers_dims) - 1):
            self.linear_relu_stack.add_module(f'linear_{i}', nn.Linear(layers_dims[i], layers_dims[i+1]))
            self.linear_relu_stack.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class FCBatchNormDropout(nn.Module):
    def __init__(self, layers_dims, dropout=0.5):
        super().__init__()
        self.type = 'FCBatchNormDropout'
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential()

        for i in range(len(layers_dims) - 1):
            self.linear_relu_stack.add_module(f'linear_{i}', nn.Linear(layers_dims[i], layers_dims[i+1]))
            self.linear_relu_stack.add_module(f'batchnorm_{i}', nn.BatchNorm1d(layers_dims[i+1]))
            self.linear_relu_stack.add_module(f'relu_{i}', nn.ReLU())
            self.linear_relu_stack.add_module(f'dropout_{i}', nn.Dropout(dropout))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CNNVanilla(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = 'CNNVanilla'
        self.flatten = nn.Flatten()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 9, stride = 2, padding = 'valid'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 7, stride = 2, padding = 'valid'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Add empty channel dimension
        x = x.unsqueeze(1)

        # Reorder the dimensions of the input tensor
        # (channels, batch_size, time_steps) -> (batch_size, channels, time_steps)
        # x = x.permute(0, 2, 1)

        x = self.conv_relu_stack(x)

        x = x.view(x.size(0), -1)
        logits = self.linear_relu_stack(x)
        return logits

class CNNDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = 'CNNDeep'
        self.flatten = nn.Flatten()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 9, padding = 'same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 7, padding = 'same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 'same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 5, padding = 'same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels = 128, out_channels = 192, kernel_size = 3, padding = 'same'),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels = 192, out_channels = 256, kernel_size = 3, padding = 'same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels = 256, out_channels = 329, kernel_size = 3, padding = 'same'),
            nn.BatchNorm1d(329),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels = 329, out_channels = 393, kernel_size = 3, padding = 'same'),
            nn.BatchNorm1d(393),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1179, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Add empty channel dimension
        x = x.unsqueeze(1)

        # Reorder the dimensions of the input tensor
        # (channels, batch_size, time_steps) -> (batch_size, channels, time_steps)
        # x = x.permute(0, 2, 1)

        x = self.conv_relu_stack(x)

        x = x.view(x.size(0), -1)
        logits = self.linear_relu_stack(x)
        return logits
