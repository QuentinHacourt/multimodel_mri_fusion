cmake_minimum_required(VERSION 3.15)
project(MRI_Fusion)

find_package(OpenCV REQUIRED)
find_package(SimpleITK REQUIRED)
find_package(Eigen3 REQUIRED)

include(${SimpleITK_USE_FILE})

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(MRI_Fusion ${SOURCES})

target_include_directories(MRI_Fusion PRIVATE ${OpenCV_INCLUDE_DIRS}
                                              ${SimpleITK_INCLUDE_DIRS})

target_link_libraries(MRI_Fusion PRIVATE ${SimpleITK_LIBRARIES} ${OpenCV_LIBS}
                                         Eigen3::Eigen)
