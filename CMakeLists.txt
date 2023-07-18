cmake_minimum_required(VERSION 3.10)
project(OPENVINO_DEMO)
set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_BUILD_TYPE RELEASE)

find_package(OpenCV 4.0 REQUIRED)
set(OpenCV_LIBS ${OpenCV_LIBS} opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)

aux_source_directory(./algorithms ALGORITHM_SRC_FILE)
add_executable(main
            ${ALGORITHM_SRC_FILE}
            main.cpp openvino_bridge.h openvino_bridge.cpp)
target_include_directories(main PUBLIC
    ~/open_vino/openvino_2022/runtime/include           # replace with your own installation location
    ~/open_vino/openvino_2022/runtime/include/ie        # replace with your own installation location
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(main 
    ~/open_vino/openvino_2022/runtime/lib/intel64/libopenvino.so    # replace with your own installation location
    ${OpenCV_LIBS}
) 