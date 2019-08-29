find_files_grouped(FILES_CPP  *.hpp* *.cpp*)
add_executable(summer-standalone
	${FILES_CPP}
)

set_target_properties(summer-standalone PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(summer-standalone
	summer-librender-cpu
	summer-librender-gpu
	summer-libcuda
	summer-liboptix
	summer-libutil
	summer-libbase
)

target_link_libraries(summer-standalone
	summer-librender-cpu
	summer-librender-gpu
	summer-libcuda
	summer-liboptix
	summer-libutil
	#summer-libbase
	${EXTERNAL_LIBRARIES}
)
