# minimum required cmake version: 3.1.0
#cmake_minimum_required(VERSION 3.1.0)
cmake_minimum_required(VERSION 2.8)
set(OpenCV_FOUND 1)



project(truckDet)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()
add_definitions("-std=c++11")

# truckDet

find_package( OpenCV REQUIRED )
set(SOURCES truckDet.cpp)
add_executable(truckDet ${SOURCES})

#target_link_libraries(truckDet${DEPENDENCIES})
target_link_libraries(truckDet ${OpenCV_LIBS})


set_target_properties (truckDet PROPERTIES
    FOLDER "Examples"
)

install(
    TARGETS
    truckDet
    RUNTIME DESTINATION
    ${CMAKE_INSTALL_PREFIX}/bin
)

