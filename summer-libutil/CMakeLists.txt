find_files_grouped(FILES_CPP  *.hpp* *.cpp*)
add_library(summer-libutil
	${FILES_CPP}
)

set_target_properties(summer-libutil PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(summer-libutil
	summer-libbase
)
