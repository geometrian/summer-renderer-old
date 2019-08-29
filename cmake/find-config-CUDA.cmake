find_package(CUDA 10 REQUIRED)

mark_as_advanced(CUDA_HOST_COMPILER)
mark_as_advanced(CUDA_SDK_ROOT_DIR)

find_program(CUDA_BIN2C_PATH bin2c REQUIRED)

add_definitions(-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
set(EXTERNAL_LIBRARIES
	${EXTERNAL_LIBRARIES}
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
)
