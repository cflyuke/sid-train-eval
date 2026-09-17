from typing import List

from modules.utils import L2NormLayer

from torch import nn
from torch import Tensor

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.0,
        normalize: bool = False
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout

        pre_dim = input_dim
        self.mlp = nn.Sequential()
        for dim in hidden_dims:
            self.mlp.append(nn.Linear(pre_dim, dim, bias=False))
            if dropout != 0:
                self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(nn.SiLU())
            pre_dim = dim
        self.mlp.append(nn.Linear(pre_dim, out_dim, bias=False))
        if normalize:
            self.mlp.append(L2NormLayer())

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.input_dim, f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)

