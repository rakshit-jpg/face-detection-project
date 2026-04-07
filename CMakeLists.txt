cmake_minimum_required(VERSION 3.10)
project(FaceDetectionPBL)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/Users/HP/Downloads/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/lib")

find_package(OpenCV REQUIRED)

add_executable(face_detection face-detection.cpp)

target_include_directories(face_detection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face_detection PRIVATE ${OpenCV_LIBS})

target_link_options(face_detection PRIVATE -mconsole)