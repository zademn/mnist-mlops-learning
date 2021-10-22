import torch
import torchvision


class LinearModel(torch.nn.Module):
    def __init__(self, hyperparameters: dict):
        super(LinearModel, self).__init__()

        # Get model config
        self.input_dim = hyperparameters['input_dim']
        self.output_dim = hyperparameters['output_dim']
        self.hidden_dims = hyperparameters['hidden_dims']
        self.negative_slope = hyperparameters.get("negative_slope", .2)

        # Create layer list
        self.layers = torch.nn.ModuleList([])
        all_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.layers.append(torch.nn.Linear(in_dim, out_dim))

        self.num_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = torch.nn.functional.leaky_relu(
                x, negative_slope=self.negative_slope)
        x = self.layers[-1](x)
        return torch.nn.functional.softmax(x, dim=-1)
