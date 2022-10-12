###########################################################################
# Setup PyTorch
###########################################################################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/python/torch_mlir/cmake/modules")
include(TorchMLIRPyTorch)

TorchMLIRProbeForPyTorchInstall()
if(TORCH_MLIR_USE_INSTALLED_PYTORCH)
  TorchMLIRConfigurePyTorch()
else()
  set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libtorch/share/cmake/Torch")
endif()

find_package(Torch 1.11 REQUIRED)

###########################################################################
# Library definition
###########################################################################

# C++ extension
include_directories(BEFORE
  ${TORCH_INCLUDE_DIRS}
)
add_library(sparse_op SHARED sparse_op.cpp)
target_link_libraries(sparse_op
  ${TORCH_LIBRARIES}
)
# Because the custom op library is a bit odd, we'd like it to stay with the
# Python component in the build directory.
set_target_properties(sparse_op PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/sparse_custom_op/"
  COMPILE_FLAGS "${TORCH_CXXFLAGS}"
)
torch_mlir_python_target_compile_options(sparse_op)
mlir_check_all_link_libraries(sparse_op)