add_library(Vertex vertex.cu)

target_include_directories(Vertex PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../triangle"
)

target_link_libraries(Vertex PRIVATE
  Object
  Vector
  Triangle
  #Model
)
