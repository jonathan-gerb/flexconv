import torch

def linspace_grid(grid_sizes):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    tensors = []
    for size in grid_sizes:
        tensors.append(torch.linspace(-1, 1, steps=size))
    grid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=0)
    return grid

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)] * n)


def pairwise(iterable):
    return grouped(iterable, 2)
