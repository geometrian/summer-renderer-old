find_files_grouped(FILES_CPP  *.hpp* *.cpp*)
add_library(summer-libcuda
	${FILES_CPP}
)

set_target_properties(summer-libcuda PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(summer-libcuda
	summer-libbase
)
