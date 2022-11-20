add_library(Vector vector.cu)

target_include_directories(Vector PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../ray"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
)

target_link_libraries(Vector PRIVATE
  Ray
  #Model
)
