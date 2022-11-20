add_library(Ray ray.cu)

target_include_directories(Ray PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../octtree"
  "${CMAKE_CURRENT_SOURCE_DIR}/../material"
  "${CMAKE_CURRENT_SOURCE_DIR}/../lights"
  "${CMAKE_CURRENT_SOURCE_DIR}/../spheres"
  "${CMAKE_CURRENT_SOURCE_DIR}/../camera"
  "${CMAKE_CURRENT_SOURCE_DIR}/../triangle"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
)

target_link_libraries(Ray PRIVATE
  Object
  Vector
  OctTree
  Material
  Lights
  Spheres
  Camera
  Triangle
  #Model
)
