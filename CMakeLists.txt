cmake_minimum_required(VERSION 3.13 FATAL_ERROR)



#Renderer

project(summer-renderer)



#CMake and generation configuration

set(cmake_helpers_dir "${PROJECT_SOURCE_DIR}/cmake/")
set(CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH} ${cmake_helpers_dir}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo" CACHE STRING "" FORCE)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#set(CMAKE_SUPPRESS_REGENERATION true)
#set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

if(MSVC)
	#Ignore warnings about `throw`ing from a destructor.  This is allowed by the C++ standard,
	#	although it must be explicitly tagged `noexcept(false)`.  We don't do that, but it's fine
	#	because we treat this situation as fatal anyway.
	add_compile_options(/wd4297)

	#Stupid macros
	add_definitions(-DNOMINMAX)
	#Reasonable character set for API calls
	add_definitions(-D_UNICODE)
endif(MSVC)



#C++ Configuration

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#	No, Microsoft, the standard library is *not* deprecated.
if(MSVC)
	add_definitions("-D_CRT_SECURE_NO_WARNINGS")
endif(MSVC)



#Renderer configuration and arguments

option(SUMMER_SUPPORT_WINDOWED "Support a windowed mode to show progress (req. GLFW)" ON)

if(SUMMER_SUPPORT_WINDOWED)
	add_definitions("-DSUMMER_SUPPORT_WINDOWED")
	set(SUMMER_WINDOW_ARG " --window")
else()
	set(SUMMER_WINDOW_ARG "")
endif()



#Find required packages

#	OptiX
include("${cmake_helpers_dir}/find-config-OptiX.cmake")

#	CUDA
include("${cmake_helpers_dir}/find-config-CUDA.cmake")

#	GLFW3
if(SUMMER_SUPPORT_WINDOWED)
	include("${cmake_helpers_dir}/find-config-GLFW3.cmake")
endif()

#	GLAD
include("${cmake_helpers_dir}/find-config-GLAD.cmake")

#	OpenGL
if(SUMMER_SUPPORT_WINDOWED)
	include("${cmake_helpers_dir}/find-config-OpenGL.cmake")
endif()

#	GLM
include("${cmake_helpers_dir}/find-config-GLM.cmake")

#	TinyGLTF
include("${cmake_helpers_dir}/find-config-TinyGLTF.cmake")

#	Threads
if (UNIX)
	include("${cmake_helpers_dir}/find-config-threads.cmake")
endif()

#	Output
message(STATUS "OptiX    at ${OptiX_INSTALL_DIR}")
message(STATUS "CUDA     at ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "GLFW3    at ${GLFW3_INCLUDE_DIR}")
message(STATUS "GLAD     at ${GLAD_INCLUDE_DIR}")
message(STATUS "GLM      at ${GLM_INCLUDE_DIR}")
message(STATUS "TinyGLTF at ${TinyGLTF_INCLUDE_DIR}")







#Renderer components

include("${cmake_helpers_dir}/helper-functions.cmake")

add_subdirectory(summer-libbase)

add_subdirectory(summer-libutil)

add_subdirectory(summer-libcuda)
add_subdirectory(summer-libopengl)
add_subdirectory(summer-liboptix)

add_subdirectory(summer-librender)

add_subdirectory(summer-standalone)



#Set startup folder for Visual Studio

set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT summer-standalone )
