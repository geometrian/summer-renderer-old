find_package(GLAD REQUIRED)

include_directories(${GLAD_INCLUDE_DIR})
set(EXTERNAL_LIBRARIES
	${EXTERNAL_LIBRARIES}
	${GLAD_LIBRARY}
)
