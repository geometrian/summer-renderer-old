find_files_grouped(FILE_CUDA_MAIN *all-kernels.cu*)
#set(CUDA_VERBOSE_BUILD ON)
add_library_embeddedptx(summer-librender-gpubuild
	${FILE_CUDA_MAIN}
	COMPILE_OPTIONS --use_fast_math
)



find_files_grouped(FILES_CPP  *.hpp* *.cpp*)
find_files_grouped(FILES_CUDA_OTHER *.cu*)
add_library(summer-librender
	${FILES_CPP}
	${FILES_CUDA_OTHER}
)

set_target_properties(summer-librender PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(summer-librender
	summer-librender-gpubuild
	summer-libcuda
	summer-liboptix
	summer-libutil
	summer-libbase
)
