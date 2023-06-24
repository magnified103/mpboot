## =================================================================================================
## Setup source files
## =================================================================================================
add_executable(mpboot
		alignment.cpp
		alignmentpairwise.cpp
		circularnetwork.cpp
		eigendecomposition.cpp
		greedy.cpp
		gss.cpp
		guidedbootstrap.cpp
		gurobiwrapper.cpp
		gzstream.cpp
		hashsplitset.cpp
		iqtree.cpp
		maalignment.cpp
		matree.cpp
		mexttree.cpp
		mpdablock.cpp
		msetsblock.cpp
		msplitsblock.cpp
		mtree.cpp
		mtreeset.cpp
		ncbitree.cpp
		ngs.cpp
		node.cpp
		optimization.cpp
		parsmultistate.cpp
		pattern.cpp
		pda.cpp
		pdnetwork.cpp
		pdtree.cpp
		pdtreeset.cpp
		phyloanalysis.cpp
		phylonode.cpp
		phylosupertree.cpp
		phylotree.cpp
		phylotreesse.cpp
		#phylotreeeigen.cpp
		pruning.cpp
		split.cpp
		splitgraph.cpp
		splitset.cpp
		stoprule.cpp
		superalignment.cpp
		superalignmentpairwise.cpp
		supernode.cpp
		tinatree.cpp
		tools.cpp
		whtest_wrapper.cpp
		lpwrapper.c
		nnisearch.cpp
		phylosupertreeplen.cpp
		phylotesting.cpp
		ecopd.cpp
		ecopdmtreeset.cpp
		graph.cpp
		candidateset.cpp
		checkpoint.cpp
		parstree.cpp
		sprparsimony.cpp
		tbrparsimony.cpp
		uppass.cpp
		test.cpp
		instrset_detect.cpp
		)

## =================================================================================================
## Compilation flags
## =================================================================================================

## Executable suffix
set(EXE_SUFFIX "")

## Configure C++ standard
target_compile_features(mpboot PRIVATE cxx_std_17)

## Include itself
target_include_directories(mpboot PRIVATE "${PROJECT_SOURCE_DIR}/src")

## Configure PLL
if(MPBOOT_USE_PLL)
	target_compile_definitions(mpboot PRIVATE USING_PLL)
	set(EXE_SUFFIX "${EXE_SUFFIX}-pll")
endif()

## Configure OpenMP
if(MPBOOT_USE_OMP)
	find_package(OpenMP REQUIRED)
	target_link_libraries(mpboot PRIVATE OpenMP::OpenMP_CXX)
	set(EXE_SUFFIX "${EXE_SUFFIX}-omp")
endif()

## Configure Highway
if(MPBOOT_HWY_TARGET STREQUAL "AVX2")
	target_compile_options(mpboot PRIVATE -march=haswell -maes)
	target_compile_definitions(mpboot PRIVATE __SSE3 __AVX)
	set(EXE_SUFFIX "${EXE_SUFFIX}-avx2")
elseif(MPBOOT_HWY_TARGET STREQUAL "SSE4")
	target_compile_options(mpboot PRIVATE -march=westmere -maes)
	target_compile_definitions(mpboot PRIVATE __SSE3)
	set(EXE_SUFFIX "${EXE_SUFFIX}-sse4")
elseif(MPBOOT_HWY_TARGET STREQUAL "NEON")
	## Nothing to do as NEON is enabled by default in aarch64
	set(EXE_SUFFIX "${EXE_SUFFIX}-neon")
else()
	target_compile_options(mpboot PRIVATE -march=native)
endif()

## Static build
#target_link_options(mpboot PRIVATE -static)

## Undef NDEBUG, which enables assertions
target_compile_options(mpboot PRIVATE -UNDEBUG)

## Configure executable suffix
set_target_properties(mpboot PROPERTIES OUTPUT_NAME "mpboot${EXE_SUFFIX}")

## =================================================================================================
## Setup dependencies
## =================================================================================================

## Eigen 3.2.10
target_include_directories(mpboot PRIVATE "${PROJECT_SOURCE_DIR}/extern/eigen")

## zlib 1.2.13
target_include_directories(mpboot PRIVATE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
target_link_libraries(mpboot PRIVATE zlib)

## vectorclass version 1
#add_subdirectory("${PROJECT_SOURCE_DIR}/extern/vectorclass" "${PROJECT_BINARY_DIR}/extern/vectorclass")
#target_link_libraries(mpboot PRIVATE vectorclass)

## Highway 1.0.4
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(HWY_ENABLE_TESTS OFF CACHE BOOL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/highway" "${PROJECT_BINARY_DIR}/extern/highway")
target_link_libraries(mpboot PRIVATE hwy)

## pllrepo
add_subdirectory(pllrepo/src)
#target_link_libraries(mpboot PRIVATE pll)

## model
add_subdirectory(model)
#target_link_libraries(mpboot PRIVATE model)

## whtest
add_subdirectory(whtest)
#target_link_libraries(mpboot PRIVATE whtest)

## ncl
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/ncl" ncl)
target_link_libraries(mpboot PRIVATE ncl)

## sprng
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/sprng" sprng)
target_link_libraries(mpboot PRIVATE sprng)

## =================================================================================================
## IQTREE configuration generator
## =================================================================================================
include(CheckSymbolExists)
check_symbol_exists(gettimeofday "sys/time.h" HAVE_GETTIMEOFDAY)
check_symbol_exists(getrusage "sys/resource.h" HAVE_GETRUSAGE)
check_symbol_exists(GlobalMemoryStatusEx "Windows.h" HAVE_GLOBALMEMORYSTATUSEX)
configure_file(
		"${PROJECT_SOURCE_DIR}/src/iqtree_config.h.in"
		"${PROJECT_BINARY_DIR}/src/iqtree_config.h"
)
target_include_directories(mpboot PRIVATE "${PROJECT_BINARY_DIR}/src")
target_compile_definitions(mpboot PRIVATE IQ_TREE)