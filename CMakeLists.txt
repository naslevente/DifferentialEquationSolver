cmake_minimum_required(VERSION 2.8)

project( Source )

add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=0)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( Source Source.cpp )
target_link_libraries(Source "${TORCH_LIBRARIES}")
# set_property(TARGET Source PROPERTY CXX_STANDARD 11)
target_link_libraries( Source ${OpenCV_LIBS} )
