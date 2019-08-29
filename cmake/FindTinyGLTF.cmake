#Default locations
set(TINYGLTF_SEARCH_DIRS
	"/usr/include/tinygltf"
	"/usr/local/include/tinygltf"

	"C:/Program Files (x86)/TinyGLTF"

	"C:/Program Files (x86)/Windows Kits/10/Lib/user/tinygltf"

	"${CMAKE_SOURCE_DIR}/include"
)

#User-specified location
if(TINYGLTF_ROOT_DIR)
	set(TINYGLTF_SEARCH_DIRS
		${TINYGLTF_ROOT_DIR}
		${TINYGLTF_SEARCH_DIRS}
	)
endif(TINYGLTF_ROOT_DIR)

#Locate header
find_path(TinyGLTF_INCLUDE_DIR "tiny_gltf.h" PATHS ${TINYGLTF_SEARCH_DIRS})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TinyGLTF DEFAULT_MSG TinyGLTF_INCLUDE_DIR)
