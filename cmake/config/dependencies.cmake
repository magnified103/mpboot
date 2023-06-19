## =================================================================================================
## Manage dependencies
## =================================================================================================
include(FetchContent)

## FetchContent_MakeAvailable but EXCLUDE_FROM_ALL
macro(FetchContent_MakeAvailableExcludeFromAll)
    foreach(contentName IN ITEMS ${ARGV})
        string(TOLOWER ${contentName} contentNameLower)
        FetchContent_GetProperties(${contentName})
        if(NOT ${contentNameLower}_POPULATED)
            FetchContent_Populate(${contentName})
            if(EXISTS ${${contentNameLower}_SOURCE_DIR}/CMakeLists.txt)
                add_subdirectory(
                        ${${contentNameLower}_SOURCE_DIR}
                        ${${contentNameLower}_BINARY_DIR}
                        EXCLUDE_FROM_ALL
                )
            endif()
        endif()
    endforeach()
endmacro()

FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG 04f42ceca40f73e2978b50e93806c2a18c1281fc    # 1.2.13
)

#FetchContent_Declare(
#        Eigen
#        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#        GIT_TAG 1c9aa054c776e16ef872cbd58cb77eef2b732891    # 3.2.10
#)

FetchContent_MakeAvailableExcludeFromAll(zlib)
