cmake_minimum_required(VERSION 3.10)
project(MRI_Fusion)

find_package(OpenCV REQUIRED)
find_package(SimpleITK REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(MRI_Fusion ${SOURCES})
# add_executable(MRI_Fusion src/main.cpp)

target_include_directories(MRI_Fusion PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(MRI_Fusion PRIVATE ${SimpleITK_LIBRARIES} ${OpenCV_LIBS})
