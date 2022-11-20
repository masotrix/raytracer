add_library(Mesh mesh.cu)
set_target_properties(Mesh PROPERTIES CUDA_ARCHITECTURES "75")

target_include_directories(Mesh PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../octtree"
  "${CMAKE_CURRENT_SOURCE_DIR}/../material"
  "${CMAKE_CURRENT_SOURCE_DIR}/../triangle"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vertex"
)

target_link_libraries(Mesh PRIVATE
  Object
  Vector
  OctTree
  Material
  Triangle
  #Model
  Vertex
)
