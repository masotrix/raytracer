add_library(Display display.cpp)

find_package(PNG REQUIRED)
#include_directories(${PNG_INCLUDE_DIR})

target_include_directories(Display PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${PNG_INCLUDE_DIR}"
)

target_link_libraries(Display PRIVATE ${PNG_LIBRARY})
