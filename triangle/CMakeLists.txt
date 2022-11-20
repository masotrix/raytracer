add_library(Triangle triangle.cu)

target_include_directories(Triangle PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../mesh"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vertex"
  "${CMAKE_CURRENT_SOURCE_DIR}/../ray"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
)

target_link_libraries(Triangle PRIVATE
  Object
  Vector
  Mesh
  Vertex
  Ray
  #Model
)
