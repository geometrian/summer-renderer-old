find_files_grouped(FILES_CPP  *.hpp* *.cpp*)
add_library(summer-libopengl
	${FILES_CPP}
)

set_target_properties(summer-libopengl PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(summer-libopengl
	summer-libbase
)
