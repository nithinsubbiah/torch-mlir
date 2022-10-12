import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
lib = os.path.join(*[current_dir, 'libsparse_op.so'])
torch.ops.load_library(lib)
