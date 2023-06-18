add_library(sprng
        lcg64.c
        makeseed.c
        memory.c
        store.c
        primes-lcg64.c
        checkid.c
        )

target_include_directories(sprng PRIVATE ${PROJECT_SOURCE_DIR}/include/sprng)
target_include_directories(sprng INTERFACE ${PROJECT_SOURCE_DIR}/include)
