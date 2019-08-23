cmake_minimum_required(VERSION 3.13 FATAL_ERROR)



#Renderer

project(summer-renderer)



#CMake and generation configuration

set(helper_cmake_dir "${PROJECT_SOURCE_DIR}/cmake/")
set(CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH}
	${helper_cmake_dir}
)
include(${helper_cmake_dir}/configure_build_type.cmake)
include(${helper_cmake_dir}/configure_optix.cmake)

set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo" CACHE STRING "" FORCE )

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#set(CMAKE_SUPPRESS_REGENERATION true)
#set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")

if (MSVC)
	#Ignore warnings about `throw`ing from a destructor.  This is allowed by the C++ standard,
	#	although it must be explicitly tagged `noexcept(false)`.  We don't do that, but it's fine
	#	because this should be fatal.
	add_compile_options(/wd4297)
endif(MSVC)



#C++ Configuration

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#	No, Microsoft, the standard library is *not* deprecated.
add_definitions("-D_CRT_SECURE_NO_WARNINGS")



#Renderer configuration and arguments

option(SUPPORT_WINDOWED "Support a windowed mode to show progress (req. GLFW)" ON)

if(SUPPORT_WINDOWED)
	add_definitions("-DSUPPORT_WINDOWED")
	set(WINDOW_ARG " --window")
else()
	set(WINDOW_ARG "")
endif()
#message(STATUS "${WINDOW_ARG}")



#Find required packages

#	CUDA
find_package(CUDA 10 REQUIRED)
message(STATUS "CUDA     at ${CUDA_TOOLKIT_ROOT_DIR}")
mark_as_advanced(CUDA_HOST_COMPILER)
mark_as_advanced(CUDA_SDK_ROOT_DIR)
mark_as_advanced(CUDA_TOOLKIT_ROOT_DIR)
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${CUDA_LIBRARIES})
set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${CUDA_CUDA_LIBRARY})

#	GLFW
if(SUPPORT_WINDOWED)
	find_package(GLFW3 REQUIRED)
	message(STATUS "GLFW3    at ${GLFW3_INCLUDE_DIR}")
	mark_as_advanced(GLFW3_DIR)
	include_directories(${GLFW3_INCLUDE_DIR})
	set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${GLFW3_LIBRARY})
endif()

#	GLM
find_package(GLM REQUIRED)
message(STATUS "GLM      at ${GLM_INCLUDE_DIR}")
include_directories(${GLM_INCLUDE_DIR})

#	GLAD
find_package(GLAD REQUIRED)
message(STATUS "GLAD     at ${GLAD_INCLUDE_DIR}")
include_directories(${GLAD_INCLUDE_DIR})
set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${GLAD_LIBRARY})

#	TinyGLTF
find_package(TinyGLTF REQUIRED)
message(STATUS "TinyGLTF at ${TinyGLTF_INCLUDE_DIR}")
include_directories(${TinyGLTF_INCLUDE_DIR})

#	OpenGL
if(SUPPORT_WINDOWED)
	find_package(OpenGL REQUIRED)
	message(STATUS "OpenGL found")
	set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${OPENGL_LIBRARIES})
	set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${OPENGL_gl_LIBRARY})
endif()

#	OptiX
find_package(OptiX 7 REQUIRED)
message(STATUS "OptiX at ${OptiX_INSTALL_DIR}")
include_directories(${OptiX_INCLUDE})
set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} ${optix_LIBRARY})

#	Threads
if (UNIX)
	set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} pthread)
endif()



#Renderer components

add_subdirectory(summer-libbase)

add_subdirectory(summer-libutil)

add_subdirectory(summer-libcuda)
add_subdirectory(summer-liboptix)

add_subdirectory(summer-librender)

add_subdirectory(summer-standalone)



#Set startup folder for Visual Studio

set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT summer-standalone )
