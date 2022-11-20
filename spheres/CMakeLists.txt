add_library(Spheres spheres.cu)
set_target_properties(Spheres PROPERTIES CUDA_ARCHITECTURES "75")

target_include_directories(Spheres PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../material"
  "${CMAKE_CURRENT_SOURCE_DIR}/../ray"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
)

target_link_libraries(Spheres PRIVATE
  Object
  Material
  Ray
  Vector
  #Model
)
