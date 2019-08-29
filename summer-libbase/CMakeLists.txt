find_files_grouped(FILES_CPP  *.hpp* *.cpp*)
add_library(summer-libbase
	${FILES_CPP}
)

set_target_properties(summer-libbase PROPERTIES LINKER_LANGUAGE CXX)
