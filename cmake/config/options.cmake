## =================================================================================================
## Handle options
## =================================================================================================
option(IQTREE_FLAGS     "Flags for MPBoot" "")

if(IQTREE_FLAGS MATCHES "pll")
    set(MPBOOT_USE_PLL ON)
else()
    set(MPBOOT_USE_PLL OFF)
endif()

if(IQTREE_FLAGS MATCHES "omp")
    set(MPBOOT_USE_OMP ON)
else()
    set(MPBOOT_USE_OMP OFF)
endif()

if(IQTREE_FLAGS MATCHES "avx2|fma")
    set(MPBOOT_HWY_TARGET "AVX2")
elseif(IQTREE_FLAGS MATCHES "avx|sse4")
    set(MPBOOT_HWY_TARGET "SSE4")
elseif(IQTREE_FLAGS MATCHES "neon")
    set(MPBOOT_HWY_TARGET "NEON")
else()
    set(MPBOOT_HWY_TARGET "")
endif()
