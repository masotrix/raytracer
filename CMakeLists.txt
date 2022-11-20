cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(CUDARayTracer LANGUAGES CXX CUDA)
add_executable(Exec main.cu ${CUDA_LIBRARY})
#set_target_properties(Exec PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

#add_subdirectory(model)
add_subdirectory(camera)
add_subdirectory(ray)
add_subdirectory(material)
add_subdirectory(lights)
add_subdirectory(vector)
add_subdirectory(object)
add_subdirectory(octtree)
add_subdirectory(mesh)
add_subdirectory(triangle)
add_subdirectory(vertex)
add_subdirectory(spheres)
add_subdirectory(parsing)
add_subdirectory(display)

target_link_libraries(Exec PRIVATE
  Display
  Parsing
  Ray
  Camera
  Lights
  Object
  Material
  Spheres
  Vector
  OctTree
  Mesh
  Triangle
  Vertex
)

target_include_directories(Exec PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/model"
  "${PROJECT_SOURCE_DIR}/material"
  "${PROJECT_SOURCE_DIR}/lights"
  "${PROJECT_SOURCE_DIR}/camera"
  "${PROJECT_SOURCE_DIR}/ray"
  "${PROJECT_SOURCE_DIR}/vector"

  "${PROJECT_SOURCE_DIR}/object"
  "${PROJECT_SOURCE_DIR}/spheres"
  "${PROJECT_SOURCE_DIR}/octtree"
  "${PROJECT_SOURCE_DIR}/mesh"
  "${PROJECT_SOURCE_DIR}/triangle"
  "${PROJECT_SOURCE_DIR}/vertex"

  "${PROJECT_SOURCE_DIR}/display"
  "${PROJECT_SOURCE_DIR}/parsing"
)
