import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_mlir
import torch_mlir.sparse_custom_op

class TopK(nn.Module):
    def forward(self, x):
        return torch.ops.sparse_op.topk(x, density=0.5)

class BlockTopK(nn.Module):
    def forward(self, x):
        return torch.ops.sparse_op.blocktopk(x, k=2, block_size=4, block_dim=1)

class Dense(nn.Module):
    def forward(self, x):
        return torch.ops.sparse_op.dense(x)

class Bernoulli(nn.Module):
    def forward(self, x):
        return torch.ops.sparse_op.bernoulli(x)

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_size):
        super().__init__()
        self.stopk = TopK()
        self.matmul = nn.Linear(input_dim, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.stopk(self.matmul.weight)
        matmul_out = self.matmul(x)
        relu_out = self.relu(matmul_out)
        return relu_out

inputs = torch.randn(2,2)
model = SimpleModel(4,4)

module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.TOSA, use_tracing=False)
print(module)
