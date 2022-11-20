add_library(Camera camera.cu)

target_include_directories(Camera PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../ray"
  "${CMAKE_CURRENT_SOURCE_DIR}/../vector"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
)

target_link_libraries(Camera PRIVATE
  Ray
  Vector
  #Model
  ${CUDA_LIBRARY}
)
