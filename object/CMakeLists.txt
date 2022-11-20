add_library(Object object.cu)
set_target_properties(Object PROPERTIES CUDA_ARCHITECTURES "75")
#set_property(TARGET Object PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(Object PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../spheres"
)

target_link_libraries(Object PRIVATE
  Spheres
  #Model
)
