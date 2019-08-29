function(find_files_grouped)
	#Usage:
	#	find_files_grouped(<file-list-to-return> <file-extensions-regexes>)

	set(GROUPED_FILEPATHS_NAME ${ARGV0})

	list(REMOVE_AT ARGV 0)
	file(GLOB_RECURSE GROUPED_FILEPATHS ${ARGV})

	foreach(FILEPATH IN ITEMS ${GROUPED_FILEPATHS})
		get_filename_component(FILEPATH_NAMEONLY "${FILEPATH}" PATH)
		string(REPLACE "${CMAKE_SOURCE_DIR}" "" PATHGROUP "${FILEPATH_NAMEONLY}")
		string(REPLACE "/" "\\" PATHGROUP "${PATHGROUP}")
		source_group("${PATHGROUP}" FILES "${FILEPATH}")
	endforeach()

	set(${GROUPED_FILEPATHS_NAME} ${GROUPED_FILEPATHS} PARENT_SCOPE)
endfunction()

function(add_library_embeddedptx)
	#Usage:
	#	add_library_embeddedptx(<library-name> <cuda-sources> [COMPILE_OPTIONS <options>])

	#Each given CUDA file is compiled into a PTX file.  Each PTX file is then stored into a string
	#	variable in a C++ file, and then all the C++ files are linked together into a new static
	#	library target, which can be linked into the final C++ program.  At runtime, the string
	#	variables can be loaded as OptiX module(s).

	#This is horrifically messy, and breaks the CUDA compilation model besides, but OptiX *only*
	#	works via PTX strings, so the only alternative would be to actually load the PTX files from
	#	actual paths on the filesystem at runtime instead , which is even messier, and also
	#	introduces the possibility of path errors.  Hopefully NVIDIA will rethink this situation in
	#	the future.

	#Parse arguments
	set(options)
	set(options_onearg COMPILE_OPTIONS)
	set(options_multiarg)
	cmake_parse_arguments(PARSE_ARGV 1 PTX "${options}" "${options_onearg}" "${options_multiarg}")
	set(LIBRARY_NAME ${ARGV0})
	set(CUDA_SOURCES ${PTX_UNPARSED_ARGUMENTS})

	#Compile each CUDA file and embed it into a C++ file (actually, a C file usable from C++; using
	#	the ".cpp" extension causes the `add_library(...)` call below to silently generate an empty
	#	target, which feels like it can't be anything other than bug with either MSVC or Cmake, but
	#	Googling for it is impossible.
	set(PTX_EMBEDDED_CPP_FILES "")
	#message(STATUS "CUDA file(s): ${CUDA_SOURCES}")
	foreach (CUDA_SOURCE ${CUDA_SOURCES})
		file(RELATIVE_PATH CUDA_SOURCE_REL ${PROJECT_SOURCE_DIR} ${CUDA_SOURCE})

		#file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/${CUDA_SOURCE_REL}.ptx"         PTX_FILE) #Cannot use due to hacky nature of `cuda_compile_ptx(...)`
		file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/${CUDA_SOURCE_REL}.ptx_embed.c" PTX_EMBEDDED_CPP_FILE)
		#message(STATUS "PTX-embedded path:     ${PTX_FILE}")
		#message(STATUS "PTX-embedded C++ path: ${PTX_EMBEDDED_CPP_FILE}")

		string(REPLACE "." "_" UNIQUE_NAME ${CUDA_SOURCE_REL})
		string(REPLACE "/" "___" UNIQUE_NAME ${UNIQUE_NAME})
		string(REPLACE "-" "_" UNIQUE_NAME ${UNIQUE_NAME})
		set(PTX_EMBEDDED_CPP_VAR  "ptx_embed___${UNIQUE_NAME}")
		#message(STATUS "PTX-embedded C++ variable: ${PTX_EMBEDDED_CPP_VAR}")

		cuda_compile_ptx(COMPILED_PTX_FILE ${CUDA_SOURCE}
			OPTIONS "${PTX_COMPILE_OPTIONS} --expt-relaxed-constexpr"#"--output-file \"${PTX_FILE}\""
		)

		add_custom_command(OUTPUT ${PTX_EMBEDDED_CPP_FILE}
			COMMAND "${CUDA_BIN2C_PATH}" -c --padd 0 --type char --name ${PTX_EMBEDDED_CPP_VAR} ${COMPILED_PTX_FILE} > ${PTX_EMBEDDED_CPP_FILE}
			DEPENDS ${COMPILED_PTX_FILE}
			COMMENT "Compiling and embedding ${CUDA_SOURCE} as ${PTX_EMBEDDED_CPP_VAR} in ${PTX_EMBEDDED_CPP_FILE}"
		)

		list(APPEND PTX_EMBEDDED_CPP_FILES ${PTX_EMBEDDED_CPP_FILE})
	endforeach()

	#message(STATUS "Library name: ${LIBRARY_NAME}")
	#message(STATUS "PTX-embedded C++ files: ${PTX_EMBEDDED_CPP_FILES}")
	add_library(${LIBRARY_NAME}
		${PTX_EMBEDDED_CPP_FILES}
	)
	set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()
