cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cppcnn)

find_package(OpenCV REQUIRED)
find_package(flashlight CONFIG REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(cppcnn main.cpp)
target_link_libraries(cppcnn PRIVATE ${OpenCV_LIBS})
target_link_libraries(cppcnn PRIVATE flashlight::flashlight)
