SET(_TINYGLTF_HEADER_SEARCH_DIRS
	"/usr/include/tinygltf"
	"/usr/local/include/tinygltf"
	"${CMAKE_SOURCE_DIR}/include"
	"C:/Program Files (x86)/TinyGLTF"
	"C:/Program Files (x86)/Windows Kits/10/Lib/user/tinygltf")

# put user specified location at beginning of search
IF (TINYGLTF_ROOT_DIR)
	SET (_TINYGLTF_HEADER_SEARCH_DIRS
		"${TINYGLTF_ROOT_DIR}"
		${_TINYGLTF_HEADER_SEARCH_DIRS}
)
ENDIF (TINYGLTF_ROOT_DIR)

# locate header
FIND_PATH(TinyGLTF_INCLUDE_DIR "tiny_gltf.h" PATHS ${_TINYGLTF_HEADER_SEARCH_DIRS})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TINYGLTF DEFAULT_MSG TINYGLTF_INCLUDE_DIR)
IF (TINYGLTF_FOUND)
	#MESSAGE(STATUS "TinyGLTF_INCLUDE_DIR = ${TinyGLTF_INCLUDE_DIR}")
ENDIF (TINYGLTF_FOUND)
