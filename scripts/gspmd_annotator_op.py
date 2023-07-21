from typing import List, Tuple

import torch
import torch.nn as nn
import torch_mlir
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

def tensor_annotate(x: torch.Tensor, k: str):
    return x

sharding_lib = torch.library.Library("sharding", "DEF")
sharding_lib.define("tensor_annotate(Tensor t, str k) -> Tensor")
sharding_lib.impl("tensor_annotate", tensor_annotate)

def sharding〇tensor_annotate〡shape(t: List[int], k: str) -> List[int]:
    return t

def sharding〇tensor_annotate〡dtype(t_rank_dtype: Tuple[int, int], k: str) -> int:
    t_rank, t_dtype = t_rank_dtype
    return t_dtype

def sharding〇tensor_annotate〡has_value_semantics() -> None:
    return

extra_library = [
    sharding〇tensor_annotate〡shape, sharding〇tensor_annotate〡dtype, sharding〇tensor_annotate〡has_value_semantics]

class CustomOpExampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 10)

    def forward(self, x):
        x = torch.ops.sharding.tensor_annotate(x, "[1,1,32]0,1,2,3,4")
        x = self.linear1(x)
        return x


mod = CustomOpExampleModule()
mod.eval()

module = torch_mlir.compile(
    mod,
    torch.ones(3, 4),
    output_type=torch_mlir.OutputType.TORCH,
    backend_legal_ops=["sharding.tensor_annotate"],
    extra_library=extra_library,
)

run_pipeline_with_repro_report(
                        module,
                        "builtin.module(func.func(canonicalize),torch-backend-to-stablehlo-backend-pipeline)",
                        description="Lowering Torch Backend IR -> StableHLO Backend IR",
                    )
import torch_mlir.ir as ir
with ir.Context() as context, ir.Location.unknown(context=context):
    module.operation.attributes[
                "mhlo.num_partitions"] = ir._i32Attr(2, context)
    module.operation.attributes[
                "mhlo.num_replicas"] = ir._i32Attr(1,context)
            
print(module)