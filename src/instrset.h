/****************************  instrset.h   **********************************
* Author:        Agner Fog
* Date created:  2012-05-30
* Last modified: 2023-06-03
* Version:       2.02.01
* Project:       vector class library
* Description:
* Header file for various compiler-specific tasks as well as common
* macros and templates. This file contains:
*
* > Selection of the supported instruction set
* > Defines compiler version macros
* > Undefines certain macros that prevent function overloading
* > Helper functions that depend on instruction set, compiler, or platform
* > Common templates for permute, blend, etc.
*
* For instructions, see vcl_manual.pdf
*
* (c) Copyright 2012-2023 Agner Fog.
* Apache License version 2.0 or later.
******************************************************************************/

#ifndef INSTRSET_H
#define INSTRSET_H 20200

// check if compiled for C++17
#if defined(_MSVC_LANG)  // MS compiler has its own version of __cplusplus with different value
#if _MSVC_LANG < 201703
#error Please compile for C++17 or higher
#endif
#else  // all other compilers
#if __cplusplus < 201703
#error Please compile for C++17 or higher
#endif
#endif

// Allow the use of floating point permute instructions on integer vectors.
// Some CPU's have an extra latency of 1 or 2 clock cycles for this, but
// it may still be faster than alternative implementations:
#define ALLOW_FP_PERMUTE  true


// Macro to indicate 64 bit mode
#if (defined(_M_AMD64) || defined(_M_X64) || defined(__amd64) ) && ! defined(__x86_64__)
#define __x86_64__ 1  // There are many different macros for this, decide on only one
#endif

// The following values of INSTRSET are currently defined:
// 2:  SSE2
// 3:  SSE3
// 4:  SSSE3
// 5:  SSE4.1
// 6:  SSE4.2
// 7:  AVX
// 8:  AVX2
// 9:  AVX512F
// 10: AVX512BW/DQ/VL
// In the future, INSTRSET = 11 may include AVX512VBMI and AVX512VBMI2, but this
// decision cannot be made before the market situation for CPUs with these
// instruction sets is better known

// Find instruction set from compiler macros if INSTRSET is not defined.
// Note: Some of these macros are not defined in Microsoft compilers
#ifndef INSTRSET
#if defined ( __AVX512VL__ ) && defined ( __AVX512BW__ ) && defined ( __AVX512DQ__ )
#define INSTRSET 10
#elif defined ( __AVX512F__ ) || defined ( __AVX512__ )
#define INSTRSET 9
#elif defined ( __AVX2__ )
#define INSTRSET 8
#elif defined ( __AVX__ )
#define INSTRSET 7
#elif defined ( __SSE4_2__ )
#define INSTRSET 6
#elif defined ( __SSE4_1__ )
#define INSTRSET 5
#elif defined ( __SSSE3__ )
#define INSTRSET 4
#elif defined ( __SSE3__ )
#define INSTRSET 3
#elif defined ( __SSE2__ ) || defined ( __x86_64__ )
#define INSTRSET 2
#elif defined ( __SSE__ )
#define INSTRSET 1
#elif defined ( _M_IX86_FP )           // Defined in MS compiler. 1: SSE, 2: SSE2
#define INSTRSET _M_IX86_FP
#else
#define INSTRSET 0
#endif // instruction set defines
#endif // INSTRSET


#if INSTRSET >= 8 && !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)
// Prevent error message in g++ and Clang when using FMA intrinsics with avx2:
#if !defined(DISABLE_WARNING_AVX2_WITHOUT_FMA)
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#endif
#elif ! defined (__clang__)
#define __FMA__  1
#endif
#endif

#if defined(__x86_64__)
// Header files for non-vector intrinsic functions including _BitScanReverse(int), __cpuid(int[4],int), _xgetbv(int)
#ifdef _MSC_VER                        // Microsoft compiler or compatible Intel compiler
#include <intrin.h>
#else
#include <x86intrin.h>                 // Gcc or Clang compiler
#endif
#endif

#include <stdint.h>                    // Define integer types with known size
#include <limits.h>                    // Define INT_MAX
#include <stdlib.h>                    // define abs(int)

int  instrset_detect(void);        // tells which instruction sets are supported
bool hasFMA3(void);                // true if FMA3 instructions supported
bool hasFMA4(void);                // true if FMA4 instructions supported
bool hasXOP(void);                 // true if XOP  instructions supported
bool hasAVX512ER(void);            // true if AVX512ER instructions supported
bool hasAVX512VBMI(void);          // true if AVX512VBMI instructions supported
bool hasAVX512VBMI2(void);         // true if AVX512VBMI2 instructions supported
bool hasF16C(void);                // true if F16C instructions supported
bool hasAVX512FP16(void);          // true if AVX512_FP16 instructions supported

// function in physical_processors.cpp:
int physicalProcessors(int * logical_processors = 0);


/*****************************************************************************
*
*    Helper functions that depend on instruction set, compiler, or platform
*
*****************************************************************************/

#if defined(__x86_64__)
// Define interface to cpuid instruction.
// input:  functionnumber = leaf (eax), ecxleaf = subleaf(ecx)
// output: output[0] = eax, output[1] = ebx, output[2] = ecx, output[3] = edx
static inline void cpuid(int output[4], int functionnumber, int ecxleaf = 0) {
#if defined(__GNUC__) || defined(__clang__)           // use inline assembly, Gnu/AT&T syntax
    int a, b, c, d;
    __asm("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(functionnumber), "c"(ecxleaf) : );
    output[0] = a;
    output[1] = b;
    output[2] = c;
    output[3] = d;

#elif defined (_MSC_VER)                              // Microsoft compiler, intrin.h included
    __cpuidex(output, functionnumber, ecxleaf);       // intrinsic function for CPUID

#else                                                 // unknown platform. try inline assembly with masm/intel syntax
    __asm {
        mov eax, functionnumber
        mov ecx, ecxleaf
        cpuid;
        mov esi, output
        mov[esi], eax
        mov[esi + 4], ebx
        mov[esi + 8], ecx
        mov[esi + 12], edx
    }
#endif
}
#endif


#endif // INSTRSET_H
