SET(_glad_HEADER_SEARCH_DIRS
	"/usr/include"
	"/usr/local/include"
	"${CMAKE_SOURCE_DIR}/include"
	"C:/Program Files (x86)/glad/include"
	"C:/Program Files (x86)/Windows Kits/10/Lib/user/glad/include"
)
SET(_glad_LIB_SEARCH_DIRS
	"/usr/lib"
	"/usr/local/lib"
	"${CMAKE_SOURCE_DIR}/lib"
	"C:/Program Files (x86)/glad/lib-msvc110"
	"C:/Program Files (x86)/Windows Kits/10/Lib/user/glad/build/x64/RelWithDebInfo"
	"C:/Program Files (x86)/Windows Kits/10/Lib/user/glad/build/x64/Release"
)

# check environment variable
SET(_glad_ENV_ROOT_DIR "$ENV{GLAD_ROOT_DIR}")
IF (NOT GLAD_ROOT_DIR AND _glad_ENV_ROOT_DIR)
	SET(GLAD_ROOT_DIR "${_glad_ENV_ROOT_DIR}")
ENDIF (NOT GLAD_ROOT_DIR AND _glad_ENV_ROOT_DIR)

# put user specified location at beginning of search
IF (GLAD_ROOT_DIR)
	list( INSERT _glad_HEADER_SEARCH_DIRS 0 "${GLAD_ROOT_DIR}/include" )
	list( INSERT _glad_LIB_SEARCH_DIRS    0 "${GLAD_ROOT_DIR}/lib"     )
ENDIF (GLAD_ROOT_DIR)

# locate header
FIND_PATH(GLAD_INCLUDE_DIR "glad/glad.h" PATHS ${_glad_HEADER_SEARCH_DIRS})
FIND_LIBRARY(GLAD_LIBRARY NAMES glad PATHS ${_glad_LIB_SEARCH_DIRS} )
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLAD DEFAULT_MSG GLAD_LIBRARY GLAD_INCLUDE_DIR)
