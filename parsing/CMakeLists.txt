add_library(Parsing parsing.cu)

target_include_directories(Parsing PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vertex"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../material"
  "${CMAKE_CURRENT_SOURCE_DIR}/../lights"
  "${CMAKE_CURRENT_SOURCE_DIR}/../spheres"
  "${CMAKE_CURRENT_SOURCE_DIR}/../camera"
  "${CMAKE_CURRENT_SOURCE_DIR}/../mesh"
  "${CMAKE_CURRENT_SOURCE_DIR}/../triangle"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
)

target_link_libraries(Parsing PRIVATE
  Object
  Vector
  Vertex
  Material
  Lights
  Spheres
  Camera
  Triangle
  Mesh
  #Model
)
