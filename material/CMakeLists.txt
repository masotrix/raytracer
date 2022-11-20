add_library(Material material.cu)

target_include_directories(Material PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../ray"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../lights"
  "${CMAKE_CURRENT_SOURCE_DIR}/../spheres"
  "${CMAKE_CURRENT_SOURCE_DIR}/../camera"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
)

target_link_libraries(Material PRIVATE
  Ray
  Object
  Vector
  Lights
  Spheres
  Camera
  #Model
)
