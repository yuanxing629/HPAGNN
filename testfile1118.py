from torch_geometric.utils import to_dense_adj
import torch

edge_index = torch.tensor([[0,0,1,2,3],[0,1,0,3,0]])
print(to_dense_adj(edge_index))