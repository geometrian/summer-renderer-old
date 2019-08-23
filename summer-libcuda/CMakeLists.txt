file(GLOB_RECURSE SOURCE_FILES *.hpp* *.cpp*)
foreach(_source_file IN ITEMS ${SOURCE_FILES})
	get_filename_component(_source_path "${_source_file}" PATH)
	string(REPLACE "${CMAKE_SOURCE_DIR}" "" _group_path "${_source_path}")
	string(REPLACE "/" "\\" _group_path "${_group_path}")
	source_group("${_group_path}" FILES "${_source_file}")
endforeach()

add_library(summer-libcuda
	${SOURCE_FILES}
)
set_target_properties(summer-libcuda PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(summer-libcuda
	summer-libbase
)
