file(GLOB_RECURSE SOURCE_FILES *.hpp* *.cpp*)
foreach(_source_file IN ITEMS ${SOURCE_FILES})
	get_filename_component(_source_path "${_source_file}" PATH)
	string(REPLACE "${CMAKE_SOURCE_DIR}" "" _group_path "${_source_path}")
	string(REPLACE "/" "\\" _group_path "${_group_path}")
	source_group("${_group_path}" FILES "${_source_file}")
endforeach()

#file(GLOB_RECURSE CUDA_FILES *.cu*)
#cuda_compile_and_embed(embedded_ptx_code ${CUDA_FILES})

add_executable(summer-standalone
	#${embedded_ptx_code}
	${SOURCE_FILES}
)
set_target_properties(summer-standalone PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(summer-standalone
	summer-librender
	summer-libcuda
	summer-liboptix
	summer-libutil
	summer-libbase
)

target_link_libraries(summer-standalone
	summer-librender
	summer-libcuda
	summer-liboptix
	summer-libutil
	#summer-libbase
	${EXTERNAL_LIBRARIES}
)
