add_library(Lights lights.cu)

target_include_directories(Lights PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../ray"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
  "${CMAKE_CURRENT_SOURCE_DIR}/../camera"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../material"
)

target_link_libraries(Lights PRIVATE
  Ray
  Camera
  Material
  Object
  Vector
  Spheres
  #Model
)
