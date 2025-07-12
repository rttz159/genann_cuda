cmake_minimum_required(VERSION 3.18)
project(genann_cuda LANGUAGES C CUDA)

add_executable(genann_cuda
  main.c
  genann.cu
  genann_cuda.cu
)

set_target_properties(genann_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 89  # Adjust for different GPU
)

target_include_directories(genann_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(genann_cuda PRIVATE CUDA::cudart m)