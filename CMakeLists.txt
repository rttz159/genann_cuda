cmake_minimum_required(VERSION 3.18)
project(genann_CUDA LANGUAGES C CUDA)

add_executable(genann_CUDA
  main.c
  genann.c
  genann_cuda.cu
)

set_target_properties(genann_CUDA PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 89  # Adjust for different GPU
)

target_include_directories(genann_CUDA PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(genann_CUDA PRIVATE CUDA::cudart m)