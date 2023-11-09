import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, no_channels):
        super().__init__()
        self.ln = torch.nn.LayerNorm(no_channels)

    def forward(self, x):
        y = self.ln(x.transpose(1, -1)).transpose(-1, 1)
        return y


class GraphBatchNorm(torch.nn.Module):
    def __init__(self, no_channels):
        super().__init__()
        self.bn = self.module = torch.nn.BatchNorm1d(no_channels)
        self.no_channels = no_channels

    def forward(self, data):
        """

        :param x: Input feature vector of size [batch, num_nodes, num_channels]. Batchnorm1d expects the
            spatial and channel dimensions in reverse ordering. Why the fuck does this work in PyG's implementation???
        """
        batch_size, num_nodes = data.x.shape[0], data.x.shape[1]
        data.x = self.bn(data.x.view(batch_size * num_nodes, -1)).view(
            batch_size, num_nodes, -1
        )
        return data
