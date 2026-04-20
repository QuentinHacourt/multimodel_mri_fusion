cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

# Find OpenCV package
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(OpenCVTest src/main.cpp)

# Link OpenCV libraries to your executable
target_link_libraries(OpenCVTest PRIVATE ${OpenCV_LIBS})

# Export compile commands for Doom Emacs (LSP)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
