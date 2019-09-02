include_directories (
	${Callbacks_SOURCE_DIR}
	${ompd-testsuite_SOURCE_DIR}
	)

find_package (OpenMP)
find_package (Threads)

list(APPEND CMAKE_C_FLAGS "${OpenMP_C_FLAGS} -g")
