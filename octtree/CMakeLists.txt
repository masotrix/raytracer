add_library(OctTree octtree.cu)

target_include_directories(OctTree PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../object"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../ray"
  "${CMAKE_CURRENT_SOURCE_DIR}/../camera"
  "${CMAKE_CURRENT_SOURCE_DIR}/../triangle"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
)

target_link_libraries(OctTree PRIVATE
  Object
  Vector
  Ray
  Camera
  Triangle
  #Model
  ${CUDA_LIBRARY}
)
