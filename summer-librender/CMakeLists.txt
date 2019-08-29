find_files_grouped(FILES_CUDA *.cu*)
#set(CUDA_VERBOSE_BUILD ON)
add_library_embeddedptx(summer-librender-gpu
	${FILES_CUDA}
	COMPILE_OPTIONS --use_fast_math
)



find_files_grouped(FILES_CPP  *.hpp* *.cpp*)
add_library(summer-librender-cpu
	${FILES_CPP}
)

set_target_properties(summer-librender-cpu PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(summer-librender-cpu
	summer-librender-gpu
	summer-libcuda
	summer-liboptix
	summer-libutil
	summer-libbase
)
