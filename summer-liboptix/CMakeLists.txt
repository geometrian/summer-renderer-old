find_files_grouped(FILES_CPP  *.hpp* *.cpp*)
add_library(summer-liboptix
	${FILES_CPP}
)

set_target_properties(summer-liboptix PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(summer-liboptix
	summer-libbase
	summer-libcuda
)
