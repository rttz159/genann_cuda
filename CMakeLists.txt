cmake_minimum_required(VERSION 3.18)
project(Genann_CUDA LANGUAGES C CUDA)

add_executable(Genann_CUDA
  main.c
  genann.c
)

set_target_properties(Genann_CUDA PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 89  # Adjust for different GPU
)

target_include_directories(Genann_CUDA PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(Genann_CUDA PRIVATE CUDA::cudart m)