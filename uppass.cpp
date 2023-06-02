/*
 * uppass.cpp
 *
 *  Created on: Apr 22, 2023
 *      Author: HynDuf
 */
#include "uppass.h"
#include "parstree.h"
#include "tools.h"
#include <algorithm>
#include <bitset>
// #include <chrono>
#include <string>
/**
 * PLL (version 1.0.0) a software library for phylogenetic inference
 * Copyright (C) 2013 Tomas Flouri and Alexandros Stamatakis
 *
 * Derived from
 * RAxML-HPC, a program for sequential and parallel estimation of phylogenetic
 * trees by Alexandros Stamatakis
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * For any other enquiries send an Email to Tomas Flouri
 * Tomas.Flouri@h-its.org
 *
 * When publishing work that uses PLL please cite PLL
 *
 * @file fastDNAparsimony.c
 */

#if defined(__MIC_NATIVE)

#include <immintrin.h>

#define VECTOR_SIZE 16
#define USHORT_PER_VECTOR 32
#define INTS_PER_VECTOR 16
#define LONG_INTS_PER_VECTOR 8
// #define LONG_INTS_PER_VECTOR (64/sizeof(long))
#define INT_TYPE __m512i
#define CAST double *
#define SET_ALL_BITS_ONE _mm512_set1_epi32(0xFFFFFFFF)
#define SET_ALL_BITS_ZERO _mm512_setzero_epi32()
#define VECTOR_LOAD _mm512_load_epi32
#define VECTOR_STORE _mm512_store_epi32
#define VECTOR_BIT_AND _mm512_and_epi32
#define VECTOR_BIT_OR _mm512_or_epi32
#define VECTOR_AND_NOT _mm512_andnot_epi32
#define VECTOR_BIT_XOR _mm512_xor_epi32

#elif defined(__AVX)

#include "vectorclass/vectorclass.h"
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#define VECTOR_SIZE 8
#define ULINT_SIZE 64
#define USHORT_PER_VECTOR 16
#define INTS_PER_VECTOR 8
#define LONG_INTS_PER_VECTOR 4
// #define LONG_INTS_PER_VECTOR (32/sizeof(long))
#define INT_TYPE __m256d
#define CAST double *
#define SET_ALL_BITS_ONE                                                       \
    (__m256d) _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, \
                               0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
#define SET_ALL_BITS_ZERO                                                      \
    (__m256d) _mm256_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000, \
                               0x00000000, 0x00000000, 0x00000000, 0x00000000)
#define VECTOR_LOAD _mm256_load_pd
#define VECTOR_BIT_AND _mm256_and_pd
#define VECTOR_BIT_OR _mm256_or_pd
#define VECTOR_STORE _mm256_store_pd
#define VECTOR_AND_NOT _mm256_andnot_pd
#define VECTOR_BIT_XOR _mm256_xor_pd

#elif (defined(__SSE3))

#include "vectorclass/vectorclass.h"
#include <pmmintrin.h>
#include <xmmintrin.h>

#define VECTOR_SIZE 4
#define USHORT_PER_VECTOR 8
#define INTS_PER_VECTOR 4
#ifdef __i386__
#define ULINT_SIZE 32
#define LONG_INTS_PER_VECTOR 4
// #define LONG_INTS_PER_VECTOR (16/sizeof(long))
#else
#define ULINT_SIZE 64
#define LONG_INTS_PER_VECTOR 2
// #define LONG_INTS_PER_VECTOR (16/sizeof(long))
#endif
#define INT_TYPE __m128i
#define CAST __m128i *
#define SET_ALL_BITS_ONE                                                       \
    _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
#define SET_ALL_BITS_ZERO                                                      \
    _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000)
#define VECTOR_LOAD _mm_load_si128
#define VECTOR_BIT_AND _mm_and_si128
#define VECTOR_BIT_OR _mm_or_si128
#define VECTOR_STORE _mm_store_si128
#define VECTOR_AND_NOT _mm_andnot_si128
#define VECTOR_BIT_XOR _mm_xor_si128

#else
// no vectorization
#define VECTOR_SIZE 1
#endif

#include "pllrepo/src/pll.h"
#include "pllrepo/src/pllInternal.h"

static pllBoolean tipHomogeneityCheckerPars(pllInstance *tr, nodeptr p,
                                            int grouping);

extern const unsigned int mask32[32];
/* vector-specific stuff */

extern double masterTime;

/* program options */
extern Params *globalParam;
static IQTree *iqtree = NULL;
unsigned int scoreTwoSubtrees;
unsigned int oldScore = 0;
static long long cnt = 0;
static node **uppass_par = NULL;
static node **branchNode = NULL;
static int *depth = NULL;
static int *distFromRmvBranch = NULL;
static bool *recalculate = NULL;
static bool *isUppassCopied = NULL;
static bool *inRadiusRange = NULL;
double total_time_uppass;
static unsigned long bestTreeScoreHits; // to count hits to bestParsimony

extern parsimonyNumber *pllCostMatrix;    // Diep: For weighted version
extern int pllCostNstates;                // Diep: For weighted version
extern parsimonyNumber *vectorCostMatrix; // BQM: vectorized cost matrix
static parsimonyNumber highest_cost;

//(if needed) split the parsimony vector into several segments to avoid overflow
// when calc rell based on vec8us
extern int pllRepsSegments;  // # of segments
extern int *pllSegmentUpper; // array of first index of the next segment, see
                             // IQTree::segment_upper
static parsimonyNumber
    *pllRemainderLowerBounds; // array of lower bound score for the
                              // un-calculated part to the right of a segment
static bool first_call =
    true; // is this the first call to pllOptimizeSprParsimony
static bool doing_stepwise_addition = false; // is the stepwise addition on
#if defined(__AVX2__)
using SimdParsTraits = SimdTraits<parsimonyNumber, 8, Vec8ui>;
#elif defined(__AVX__)
/*
 * AVX platforms like Sandy Bridge / Ivy Bridge support 256bit floating-point vector operations,
 * so it's not bad to utilize the bitwise operations that come equipped with it
 */
using SimdParsTraits = SimdTraits<parsimonyNumber, 8, Vec4d>;
#elif defined(__SSE4_2__)
using SimdParsTraits = SimdTraits<parsimonyNumber, 4, Vec4ui>;
#else
/*
 * TODO: add case for ARM NEON
 */
using SimdParsTraits = SimdTraits<parsimonyNumber, 1, std::uint32_t>;
#endif

template<> SimdParsTraits::simd_type SimdProxy<SimdParsTraits>::load() {
#if defined(__AVX__)
    return Vec4d().load((double *)array);
#elif defined(__SSE4_2__)
    return SimdParsTraits::simd_type().load((std::uint32_t *)array);
#else
    return *((SimdParsTraits::simd_type *)array);
#endif
}

template<> void SimdProxy<SimdParsTraits>::store(SimdParsTraits::simd_type vec) {
#if defined(__AVX__)
    vec.store((double *)array);
#elif defined(__SSE4_2__)
    vec.store((std::uint32_t *)array);
#else
    *array = vec;
#endif
}

template<> SimdParsTraits::simd_type SimdParsTraits::zeros() {
#if defined(__AVX__)
    return Vec4d(_mm256_set1_epi32(0x00000000));
#else
    return SimdParsTraits::simd_type(0);
#endif
}

template<> SimdParsTraits::simd_type SimdParsTraits::ones() {
#if defined(__AVX__)
    return Vec4d(_mm256_set1_epi32(0xffffffff));
#else
    return SimdParsTraits::simd_type(0xffffffff);
#endif
}

constexpr std::size_t BLOCK_SIZE = INTS_PER_VECTOR;

inline constexpr std::size_t table_index(std::size_t width, std::size_t states, std::size_t node_id,
                                         std::size_t state_id, std::size_t col_id) {
    return states * width * node_id + states * (col_id / BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE * state_id + (col_id % BLOCK_SIZE);
}

inline constexpr std::size_t table_index_divisible(std::size_t width, std::size_t states, std::size_t node_id,
                                                   std::size_t state_id, std::size_t col_id) {
    return states * width * node_id + states * col_id + BLOCK_SIZE * state_id;
}

inline constexpr std::size_t next_column_block(std::size_t states, std::size_t pos) {
    return pos + BLOCK_SIZE * states;
}

inline constexpr std::size_t next_column_simd(std::size_t states, std::size_t pos) {
    return next_column_block(states, pos);
}

inline constexpr std::size_t next_state(std::size_t pos, std::size_t step = 1) {
    return pos + BLOCK_SIZE * step;
}

/**
 * Tranform the table schema from (node x state x column) to the designated representation
 * @tparam T
 * @param table
 */
template <typename T>
inline void table_transform(T* table, int nodes, int states, int width) {
    std::vector<T> tmp(table, table + (nodes * states * width));
    for (int node = 0; node < nodes; node++) {
        for (int state = 0; state < states; state++) {
            for (int col = 0; col < width; col++) {
                table[table_index(width, states, node, state, col)] = tmp[node * states * width + state * width + col];
            }
        }
    }
}

/********************************DNA FUNCTIONS
 * *****************************************************************/

// Diep:
// store per site score to nodeNumber
#if (defined(__SSE3) || defined(__AVX))
template <typename Traits>
static inline void storePerSiteNodeScores(partitionList *pr, int model,
                                          typename Traits::simd_type v, unsigned int offset,
                                          int nodeNumber) {

    std::uint64_t counts[Traits::template size<std::uint64_t>()] __attribute__((aligned(PLL_BYTE_ALIGNMENT)));
    parsimonyNumber *buf;

    SimdProxy<Traits>(counts).store(v);

    int partialParsLength = pr->partitionData[model]->parsimonyLength * PLL_PCF;
    int nodeStart = partialParsLength * nodeNumber;
    int nodeStartPlusOffset = nodeStart + offset * PLL_PCF;
    for (std::size_t i = 0; i < Traits::template size<std::uint64_t>(); i++) {
        buf = &(pr->partitionData[model]->perSitePartialPars[nodeStartPlusOffset]);
        nodeStartPlusOffset += 64;
        //		buf =
        //&(pr->partitionData[model]->perSitePartialPars[nodeStart +
        // offset * PLL_PCF + i * ULINT_SIZE]); // Diep's 		buf =
        //&(pr->partitionData[model]->perSitePartialPars[nodeStart + offset *
        // PLL_PCF + i]); // Tomas's code
        for (std::size_t j = 0; j < 64; ++j)
            buf[j] += ((counts[i] >> j) & 1);
    }
}

// Diep:
// Add site scores in q and r to p
// q and r are children of p
template <class VectorClass>
static void addPerSiteSubtreeScoresSIMD(partitionList *pr, int pNumber,
                                        int qNumber, int rNumber) {
    assert(VectorClass::size() == INTS_PER_VECTOR);
    parsimonyNumber *pBuf, *qBuf, *rBuf;
    for (int i = 0; i < pr->numberOfPartitions; ++i) {
        int partialParsLength = pr->partitionData[i]->parsimonyLength * PLL_PCF;
        pBuf = &(pr->partitionData[i]
                     ->perSitePartialPars[partialParsLength * pNumber]);
        qBuf = &(pr->partitionData[i]
                     ->perSitePartialPars[partialParsLength * qNumber]);
        rBuf = &(pr->partitionData[i]
                     ->perSitePartialPars[partialParsLength * rNumber]);
        for (int k = 0; k < partialParsLength; k += VectorClass::size()) {
            VectorClass *pBufVC = (VectorClass *)&pBuf[k];
            VectorClass *qBufVC = (VectorClass *)&qBuf[k];
            VectorClass *rBufVC = (VectorClass *)&rBuf[k];
            *pBufVC += *qBufVC + *rBufVC;
        }
    }
}

// Diep:
// Add site scores in q and r to p
// q and r are children of p
static void addPerSiteSubtreeScores(partitionList *pr, int pNumber, int qNumber,
                                    int rNumber) {
    //	parsimonyNumber * pBuf, * qBuf, *rBuf;
    //	for(int i = 0; i < pr->numberOfPartitions; ++i){
    //		int partialParsLength = pr->partitionData[i]->parsimonyLength *
    // PLL_PCF; 		pBuf =
    // &(pr->partitionData[i]->perSitePartialPars[partialParsLength
    //* pNumber]); 		qBuf =
    //&(pr->partitionData[i]->perSitePartialPars[partialParsLength * qNumber]);
    //		rBuf =
    //&(pr->partitionData[i]->perSitePartialPars[partialParsLength
    //* rNumber]); 		for(int k = 0; k < partialParsLength; ++k)
    // pBuf[k] += qBuf[k] + rBuf[k];
    //	}

#ifdef __AVX
    addPerSiteSubtreeScoresSIMD<Vec8ui>(pr, pNumber, qNumber, rNumber);
#else
    addPerSiteSubtreeScoresSIMD<Vec4ui>(pr, pNumber, qNumber, rNumber);
#endif
}

// Diep:
// Reset site scores of p
static void resetPerSiteNodeScores(partitionList *pr, int pNumber) {
    parsimonyNumber *pBuf;
    for (int i = 0; i < pr->numberOfPartitions; ++i) {
        int partialParsLength = pr->partitionData[i]->parsimonyLength * PLL_PCF;
        pBuf = &(pr->partitionData[i]
                     ->perSitePartialPars[partialParsLength * pNumber]);
        memset(pBuf, 0, partialParsLength * sizeof(parsimonyNumber));
    }
}
#endif

/*
 * Diep: copy new version from Tomas's code for site pars
 * Here, informative site == variant site
 * IMPORTANT: 	If this function changes the definition for 'informative site'
 * as in the below comment the function of compressSankoffDNA needs revising
 */
/* check whether site contains at least 2 different letters, i.e.
   whether it will generate a score */
static pllBoolean isInformative(pllInstance *tr, int dataType, int site) {
    if (globalParam && !globalParam->sort_alignment)
        return PLL_TRUE; // because of the sync between IQTree and PLL alignment
                         // (to get correct freq of pattern)

    int informativeCounter = 0, check[256], j,
        undetermined = getUndetermined(dataType);

    const unsigned int *bitVector = getBitVector(dataType);

    unsigned char nucleotide;

    for (j = 0; j < 256; j++)
        check[j] = 0;

    for (j = 1; j <= tr->mxtips; j++) {
        nucleotide = tr->yVector[j][site];
        check[nucleotide] = 1;
        assert(bitVector[nucleotide] > 0);
    }

    for (j = 0; j < undetermined; j++) {
        if (check[j] > 0)
            informativeCounter++;
    }

    if (informativeCounter > 1)
        return PLL_TRUE;

    return PLL_FALSE;
}
static void determineUninformativeSites(pllInstance *tr, partitionList *pr,
                                        int *informative) {
    int model, number = 0, i;

    /*
       Not all characters are useful in constructing a parsimony tree.
       Invariant characters, those that have the same state in all taxa,
       are obviously useless and are ignored by the method. Characters in
       which a state occurs in only one taxon are also ignored.
       All these characters are called parsimony uninformative.

       Alternative definition: informative columns contain at least two types
       of nucleotides, and each nucleotide must appear at least twice in each
       column. Kind of a pain if we intend to check for this when using, e.g.,
       amibiguous DNA encoding.
    */

    for (model = 0; model < pr->numberOfPartitions; ++model) {

        for (i = pr->partitionData[model]->lower;
             i < pr->partitionData[model]->upper; ++i) {
            if (isInformative(tr, pr->partitionData[model]->dataType, i)) {
                informative[i] = 1;
            } else {
                informative[i] = 0;
            }
        }
    }

    /* printf("Uninformative Patterns: %d\n", number); */
}
static int pllSaveCurrentTreeSprParsimony(pllInstance *tr, partitionList *pr,
                                          int cur_search_pars) {
    iqtree->saveCurrentTree(-cur_search_pars);
    return (int)(cur_search_pars);
}

template <class Numeric, const int VECSIZE>
static void compressSankoffDNA(pllInstance *tr, partitionList *pr,
                               int *informative, int perSiteScores) {
    //	cout << "Begin compressSankoffDNA()" << endl;
    size_t totalNodes, i, model;

    totalNodes = 2 * (size_t)tr->mxtips;

    for (model = 0; model < (size_t)pr->numberOfPartitions; ++model) {
        size_t k, states = (size_t)pr->partitionData[model]->states,
                  compressedEntries, compressedEntriesPadded, entries = 0,
                  lower = pr->partitionData[model]->lower,
                  upper = pr->partitionData[model]->upper;

        //      parsimonyNumber
        //        **compressedTips = (parsimonyNumber **)rax_malloc(states *
        //        sizeof(parsimonyNumber*)), *compressedValues =
        //        (parsimonyNumber
        //        *)rax_malloc(states * sizeof(parsimonyNumber));

        for (i = lower; i < upper; ++i)
            if (informative[i])
                entries++; // Diep: here,entries counts # informative pattern

        // number of informative site patterns
        compressedEntries = entries;

#if (defined(__SSE3) || defined(__AVX))
        if (compressedEntries % VECSIZE != 0)
            compressedEntriesPadded =
                compressedEntries + (VECSIZE - (compressedEntries % VECSIZE));
        else
            compressedEntriesPadded = compressedEntries;
#else
        compressedEntriesPadded = compressedEntries;
#endif

        // parsVect stores cost for each node by state at each pattern
        // for a certain node of DNA: ptn1_A, ptn2_A, ptn3_A,..., ptn1_C,
        // ptn2_C, ptn3_C,...,ptn1_G, ptn2_G, ptn3_G,...,ptn1_T, ptn2_T,
        // ptn3_T,..., (not 100% sure) this is also the perSitePartialPars

        rax_posix_memalign((void **)&(pr->partitionData[model]->parsVect),
                           PLL_BYTE_ALIGNMENT,
                           (size_t)compressedEntriesPadded * states *
                               totalNodes * sizeof(parsimonyNumber));
        memset(pr->partitionData[model]->parsVect, 0,
               compressedEntriesPadded * states * totalNodes *
                   sizeof(parsimonyNumber));

        rax_posix_memalign(
            (void **)&(pr->partitionData[model]->informativePtnWgt),
            PLL_BYTE_ALIGNMENT,
            (size_t)compressedEntriesPadded * sizeof(Numeric));

        memset(pr->partitionData[model]->informativePtnWgt, 0,
               (size_t)compressedEntriesPadded * sizeof(Numeric));

        if (perSiteScores) {
            rax_posix_memalign(
                (void **)&(pr->partitionData[model]->informativePtnScore),
                PLL_BYTE_ALIGNMENT,
                (size_t)compressedEntriesPadded * sizeof(Numeric));
            memset(pr->partitionData[model]->informativePtnScore, 0,
                   (size_t)compressedEntriesPadded * sizeof(Numeric));
        }

        //      if (perSiteScores)
        //       {
        //         /* for per site parsimony score at each node */
        //         rax_posix_memalign ((void **)
        //         &(pr->partitionData[model]->perSitePartialPars),
        //         PLL_BYTE_ALIGNMENT, totalNodes *
        //         (size_t)compressedEntriesPadded
        //         * PLL_PCF * sizeof (parsimonyNumber)); for (i = 0; i <
        //         totalNodes
        //         * (size_t)compressedEntriesPadded * PLL_PCF; ++i)
        //        	 pr->partitionData[model]->perSitePartialPars[i] = 0;
        //       }

        // Diep: For each leaf
        for (i = 0; i < (size_t)tr->mxtips; ++i) {
            size_t w = 0, compressedIndex = 0, compressedCounter = 0, index = 0,
                   informativeIndex = 0;

            //          for(k = 0; k < states; ++k)
            //            {
            //              compressedTips[k] =
            //              &(pr->partitionData[model]->parsVect[(compressedEntriesPadded
            //              * states * (i + 1)) + (compressedEntriesPadded *
            //              k)]); compressedValues[k] = INT_MAX; // Diep
            //            }

            Numeric *tipVect =
                (Numeric *)&pr->partitionData[model]
                    ->parsVect[(compressedEntriesPadded * states * (i + 1))];

            Numeric *ptnWgt =
                (Numeric *)pr->partitionData[model]->informativePtnWgt;
            // for each informative pattern
            for (index = lower; index < (size_t)upper; index++) {

                if (informative[index]) {
                    //                	cout << "index = " << index << endl;
                    const unsigned int *bitValue = getBitVector(
                        pr->partitionData[model]
                            ->dataType); // Diep: bitValue is for dataType

                    parsimonyNumber value = bitValue[tr->yVector[i + 1][index]];

                    /*
                            memory for score per node, assuming
                       VectorClass::size()=2, and states=4 (A,C,G,T) in block of
                       size VectorClass::size()*states

                            Index  0  1  2  3  4  5  6  7  8  9  10 ...
                            Site   0  1  0  1  0  1  0  1  2  3   2 ...
                            State  A  A  C  C  G  G  T  T  A  A   C ...

                    */

                    for (k = 0; k < states; ++k) {
                        if (value & mask32[k])
                            tipVect[k * VECSIZE] =
                                0; // Diep: if the state is present,
                                   // corresponding value is set to zero
                        else
                            tipVect[k * VECSIZE] = highest_cost;
                        //					  compressedTips[k][informativeIndex]
                        //= compressedValues[k]; // Diep
                        // cout << "compressedValues[k]: " <<
                        // compressedValues[k] << endl;
                    }
                    ptnWgt[informativeIndex] = tr->aliaswgt[index];
                    informativeIndex++;

                    tipVect += 1; // process to the next site

                    // jump to the next block
                    if (informativeIndex % VECSIZE == 0)
                        tipVect += VECSIZE * (states - 1);
                }
            }

            // dummy values for the last padded entries
            for (index = informativeIndex; index < compressedEntriesPadded;
                 index++) {

                for (k = 0; k < states; ++k) {
                    tipVect[k * VECSIZE] = 0;
                }
                tipVect += 1;
            }
        }

#if (defined(__SSE3) || defined(__AVX))
        pr->partitionData[model]->parsimonyLength = compressedEntriesPadded;
#else
        pr->partitionData[model]->parsimonyLength =
            compressedEntries; // for unvectorized version
#endif
        //	cout << "compressedEntries = " << compressedEntries << endl;
        //      rax_free(compressedTips);
        //      rax_free(compressedValues);
    }

    // TODO: remove this for Sankoff?

    rax_posix_memalign((void **)&(tr->parsimonyScore), PLL_BYTE_ALIGNMENT,
                       sizeof(unsigned int) * totalNodes);

    for (i = 0; i < totalNodes; ++i)
        tr->parsimonyScore[i] = 0;

    if ((!perSiteScores) && pllRepsSegments > 1) {
        // compute lower-bound if not currently extracting per site score AND
        // having > 1 segments
        pllRemainderLowerBounds =
            new parsimonyNumber[pllRepsSegments -
                                1]; // last segment does not need lower bound
        assert(iqtree != NULL);
        int partitionId = 0;
        int ptn;
        int nptn = iqtree->aln->n_informative_patterns;
        int *min_ptn_pars = new int[nptn];

        for (ptn = 0; ptn < nptn; ptn++)
            min_ptn_pars[ptn] =
                dynamic_cast<ParsTree *>(iqtree)->findMstScore(ptn);

        for (int seg = 0; seg < pllRepsSegments - 1; seg++) {
            pllRemainderLowerBounds[seg] = 0;
            for (ptn = pllSegmentUpper[seg]; ptn < nptn; ptn++) {
                pllRemainderLowerBounds[seg] +=
                    min_ptn_pars[ptn] *
                    pr->partitionData[partitionId]->informativePtnWgt[ptn];
            }
        }

        delete[] min_ptn_pars;
    } else
        pllRemainderLowerBounds = NULL;
}
/**
 * Uppass algorithm
 */
static void compressDNAUppass(pllInstance *tr, partitionList *pr,
                              int *informative, int perSiteScores) {
    // cout << "Begin compressDNAUppass\n";
    if (pllCostMatrix != NULL) {
        if (globalParam->sankoff_short_int)
            return compressSankoffDNA<parsimonyNumberShort, USHORT_PER_VECTOR>(
                tr, pr, informative, perSiteScores);
        else
            return compressSankoffDNA<parsimonyNumber, INTS_PER_VECTOR>(
                tr, pr, informative, perSiteScores);
    }

    size_t totalNodes, i, model;

    totalNodes = 2 * (size_t)tr->mxtips;

    for (model = 0; model < (size_t)pr->numberOfPartitions; ++model) {
        size_t k, states = (size_t)pr->partitionData[model]->states,
                  compressedEntries, compressedEntriesPadded, entries = 0,
                  lower = pr->partitionData[model]->lower,
                  upper = pr->partitionData[model]->upper;

        parsimonyNumber **compressedTips = (parsimonyNumber **)rax_malloc(
                            states * sizeof(parsimonyNumber *)),
                        *compressedValues = (parsimonyNumber *)rax_malloc(
                            states * sizeof(parsimonyNumber));

        pr->partitionData[model]->numInformativePatterns =
            0; // to fix score bug THAT too many uninformative sites cause
               // out-of-bound array access

        for (i = lower; i < upper; ++i)
            if (informative[i]) {
                entries += (size_t)tr->aliaswgt[i];
                pr->partitionData[model]->numInformativePatterns++;
            }

        // cout << "entries = " << entries << '\n';
        compressedEntries = entries / PLL_PCF;

        if (entries % PLL_PCF != 0)
            compressedEntries++;
            // cout << "compressedEntries = " << compressedEntries << '\n';
            // assert(compressedEntries % 8 != 0);
#if (defined(__SSE3) || defined(__AVX))
        if (compressedEntries % INTS_PER_VECTOR != 0)
            compressedEntriesPadded =
                compressedEntries +
                (INTS_PER_VECTOR - (compressedEntries % INTS_PER_VECTOR));
        else
            compressedEntriesPadded = compressedEntries;
#else
        compressedEntriesPadded = compressedEntries;
#endif

        // cout << "parsVect\n";
        rax_posix_memalign((void **)&(pr->partitionData[model]->parsVect),
                           PLL_BYTE_ALIGNMENT,
                           (size_t)compressedEntriesPadded * states *
                               totalNodes * sizeof(parsimonyNumber));

        for (i = 0; i < compressedEntriesPadded * states * totalNodes; ++i)
            pr->partitionData[model]->parsVect[i] = 0;

        // cout << "scoreIncreaseparsVectUppass\n";
        rax_posix_memalign((void **)&(pr->partitionData[model]->parsVectUppass),
                           PLL_BYTE_ALIGNMENT,
                           (size_t)compressedEntriesPadded * states *
                               totalNodes * sizeof(parsimonyNumber));

        for (i = 0; i < compressedEntriesPadded * states * totalNodes; ++i)
            pr->partitionData[model]->parsVectUppass[i] = 0;

        // cout << "parsVectUppassLocal\n";
        rax_posix_memalign(
            (void **)&(pr->partitionData[model]->parsVectUppassLocal),
            PLL_BYTE_ALIGNMENT,
            (size_t)compressedEntriesPadded * states * totalNodes *
                sizeof(parsimonyNumber));

        for (i = 0; i < compressedEntriesPadded * states * totalNodes; ++i)
            pr->partitionData[model]->parsVectUppassLocal[i] = 0;

        rax_posix_memalign(
            (void **)&(pr->partitionData[model]->branchVectUppass),
            PLL_BYTE_ALIGNMENT,
            (size_t)compressedEntriesPadded * states * totalNodes *
                sizeof(parsimonyNumber));

        for (i = 0; i < compressedEntriesPadded * states * totalNodes; ++i)
            pr->partitionData[model]->branchVectUppass[i] = 0;

        // cout << "scoreIncrease\n";
        rax_posix_memalign((void **)&(pr->partitionData[model]->scoreIncrease),
                           PLL_BYTE_ALIGNMENT,
                           (size_t)compressedEntriesPadded * totalNodes *
                               sizeof(parsimonyNumber));
        for (i = 0; i < compressedEntriesPadded * totalNodes; ++i)
            pr->partitionData[model]->scoreIncrease[i] = 0;

        if (perSiteScores) {
            /* for per site parsimony score at each node */
            rax_posix_memalign(
                (void **)&(pr->partitionData[model]->perSitePartialPars),
                PLL_BYTE_ALIGNMENT,
                totalNodes * (size_t)compressedEntriesPadded * PLL_PCF *
                    sizeof(parsimonyNumber));
            for (i = 0;
                 i < totalNodes * (size_t)compressedEntriesPadded * PLL_PCF;
                 ++i)
                pr->partitionData[model]->perSitePartialPars[i] = 0;
        }

        for (i = 0; i < (size_t)tr->mxtips; ++i) {
            size_t w = 0, compressedIndex = 0, compressedCounter = 0, index = 0;

            for (k = 0; k < states; ++k) {
                compressedTips[k] =
                    &(pr->partitionData[model]
                          ->parsVect[(compressedEntriesPadded * states *
                                      (i + 1)) +
                                     (compressedEntriesPadded * k)]);
                compressedValues[k] = 0;
            }

            for (index = lower; index < (size_t)upper; index++) {
                if (informative[index]) {
                    const unsigned int *bitValue =
                        getBitVector(pr->partitionData[model]->dataType);

                    parsimonyNumber value = bitValue[tr->yVector[i + 1][index]];

                    for (w = 0; w < (size_t)tr->aliaswgt[index]; w++) {
                        for (k = 0; k < states; ++k) {
                            if (value & mask32[k])
                                compressedValues[k] |=
                                    mask32[compressedCounter];
                        }

                        compressedCounter++;

                        if (compressedCounter == PLL_PCF) {
                            for (k = 0; k < states; ++k) {
                                compressedTips[k][compressedIndex] =
                                    compressedValues[k];
                                compressedValues[k] = 0;
                            }

                            compressedCounter = 0;
                            compressedIndex++;
                        }
                    }
                }
            }

            for (; compressedIndex < compressedEntriesPadded;
                 compressedIndex++) {
                for (; compressedCounter < PLL_PCF; compressedCounter++)
                    for (k = 0; k < states; ++k)
                        compressedValues[k] |= mask32[compressedCounter];

                for (k = 0; k < states; ++k) {
                    compressedTips[k][compressedIndex] = compressedValues[k];
                    compressedValues[k] = 0;
                }

                compressedCounter = 0;
            }
        }
        /**
         * NOTE: Don't need this anymore, as uppass of leaves are calculated
         * normally like internal nodes
         */
        /*
        // Copy leaves' downpass to uppass
        for (i = 0;
             i < compressedEntriesPadded * states * ((size_t)tr->mxtips + 1);
             ++i) {
            pr->partitionData[model]->parsVectUppass[i] =
                pr->partitionData[model]->parsVect[i];
        }
        */

        pr->partitionData[model]->parsimonyLength = compressedEntriesPadded;

        rax_free(compressedTips);
        rax_free(compressedValues);

        // table transforms
        table_transform(pr->partitionData[model]->parsVect, totalNodes, states, compressedEntriesPadded);
        table_transform(pr->partitionData[model]->parsVectUppass, totalNodes, states, compressedEntriesPadded);
        table_transform(pr->partitionData[model]->parsVectUppassLocal, totalNodes, states, compressedEntriesPadded);
        table_transform(pr->partitionData[model]->branchVectUppass, totalNodes, states, compressedEntriesPadded);
    }

    rax_posix_memalign((void **)&(tr->parsimonyScore), PLL_BYTE_ALIGNMENT,
                       sizeof(unsigned int) * totalNodes);

    for (i = 0; i < totalNodes; ++i)
        tr->parsimonyScore[i] = 0;
    // cout << "End compressDNAUppass\n";
}

static void _updateInternalPllOnRatchet(pllInstance *tr, partitionList *pr) {
    //	cout << "lower = " << pr->partitionData[0]->lower << ", upper = " <<
    // pr->partitionData[0]->upper << ", aln->size() = " << iqtree->aln->size()
    // << endl;
    for (int i = 0; i < pr->numberOfPartitions; ++i) {
        for (int ptn = pr->partitionData[i]->lower;
             ptn < pr->partitionData[i]->upper; ptn++) {
            tr->aliaswgt[ptn] = iqtree->aln->at(ptn).frequency;
        }
    }
}
void _allocateParsimonyDataStructuresUppass(pllInstance *tr, partitionList *pr,
                                            int perSiteScores) {
    // cout << "Begin _allocateParsimonyDataStructuresUppass\n";
    int i;
    int *informative =
        (int *)rax_malloc(sizeof(int) * (size_t)tr->originalCrunchedLength);
    determineUninformativeSites(tr, pr, informative);

    if (pllCostMatrix) {
        for (int i = 0; i < pr->numberOfPartitions; ++i) {
            pr->partitionData[i]->informativePtnWgt = NULL;
            pr->partitionData[i]->informativePtnScore = NULL;
        }
    }

    compressDNAUppass(tr, pr, informative, perSiteScores);

    if (uppass_par == NULL) {
//        uppass_par = new nodeptr[tr->mxtips + tr->mxtips - 1];
        rax_posix_memalign((void**)&uppass_par, PLL_BYTE_ALIGNMENT, sizeof(nodeptr) * (tr->mxtips + tr->mxtips - 1));
        for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
            uppass_par[i] = NULL;
        }
    }
    if (branchNode == NULL) {
//        branchNode = new nodeptr[tr->mxtips + tr->mxtips - 1];
        rax_posix_memalign((void**)&branchNode, PLL_BYTE_ALIGNMENT, sizeof(nodeptr) * (tr->mxtips + tr->mxtips - 1));
        for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
            branchNode[i] = NULL;
        }
    }
    if (depth == NULL) {
//        depth = new int[tr->mxtips + tr->mxtips - 1];
        rax_posix_memalign((void**)&depth, PLL_BYTE_ALIGNMENT, sizeof(int) * (tr->mxtips + tr->mxtips - 1));
    }
    if (distFromRmvBranch == NULL) {
//        distFromRmvBranch = new int[tr->mxtips + tr->mxtips - 1];
        rax_posix_memalign((void**)&distFromRmvBranch, PLL_BYTE_ALIGNMENT, sizeof(int) * (tr->mxtips + tr->mxtips - 1));
    }
    if (recalculate == NULL) {
//        recalculate = new bool[tr->mxtips + tr->mxtips - 1];
        rax_posix_memalign((void**)&recalculate, PLL_BYTE_ALIGNMENT, sizeof(bool) * (tr->mxtips + tr->mxtips - 1));
        for (i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
            recalculate[i] = false;
        }
    }
    if (isUppassCopied == NULL) {
//        isUppassCopied = new bool[tr->mxtips + tr->mxtips - 1];
        rax_posix_memalign((void**)&isUppassCopied, PLL_BYTE_ALIGNMENT, sizeof(bool) * (tr->mxtips + tr->mxtips - 1));
        for (i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
            isUppassCopied[i] = false;
        }
    }
    if (inRadiusRange == NULL) {
//        inRadiusRange = new bool[tr->mxtips + tr->mxtips - 1];
        rax_posix_memalign((void**)&inRadiusRange, PLL_BYTE_ALIGNMENT, sizeof(bool) * (tr->mxtips + tr->mxtips - 1));
        for (i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
            inRadiusRange[i] = false;
        }
    }
    for (i = tr->mxtips + 1; i <= tr->mxtips + tr->mxtips - 1; ++i) {
        nodeptr p = tr->nodep[i];

        p->xPars = 1;
        p->next->xPars = 0;
        p->next->next->xPars = 0;
    }

    tr->ti = (int *)rax_malloc(sizeof(int) * 4 * 2 * (size_t)tr->mxtips);

    rax_free(informative);
    // cout << "End _allocateParsimonyDataStructuresUppass\n";
}

void _pllFreeParsimonyDataStructuresUppass(pllInstance *tr, partitionList *pr) {
    // cout << "Begin _pllFreeParsimonyDataStructuresUppass\n";
    size_t model;

    if (tr->parsimonyScore != NULL) {
        rax_free(tr->parsimonyScore);
        tr->parsimonyScore = NULL;
    }

    for (model = 0; model < (size_t)pr->numberOfPartitions; ++model) {
        if (pr->partitionData[model]->parsVect != NULL) {
            rax_free(pr->partitionData[model]->parsVect);
            pr->partitionData[model]->parsVect = NULL;
        }
        if (pr->partitionData[model]->parsVectUppass != NULL) {
            rax_free(pr->partitionData[model]->parsVectUppass);
            pr->partitionData[model]->parsVectUppass = NULL;
        }
        if (pr->partitionData[model]->parsVectUppassLocal != NULL) {
            rax_free(pr->partitionData[model]->parsVectUppassLocal);
            pr->partitionData[model]->parsVectUppassLocal = NULL;
        }
        if (pr->partitionData[model]->branchVectUppass != NULL) {
            rax_free(pr->partitionData[model]->branchVectUppass);
            pr->partitionData[model]->branchVectUppass = NULL;
        }
        if (pr->partitionData[model]->scoreIncrease != NULL) {
            rax_free(pr->partitionData[model]->scoreIncrease);
            pr->partitionData[model]->scoreIncrease = NULL;
        }
        if (pr->partitionData[model]->perSitePartialPars != NULL) {
            rax_free(pr->partitionData[model]->perSitePartialPars);
            pr->partitionData[model]->perSitePartialPars = NULL;
        }
    }

    if (tr->ti != NULL) {
        rax_free(tr->ti);
        tr->ti = NULL;
    }
    if (pllCostMatrix) {
        for (int i = 0; i < pr->numberOfPartitions; ++i) {
            if (pr->partitionData[i]->informativePtnWgt != NULL) {
                rax_free(pr->partitionData[i]->informativePtnWgt);
                pr->partitionData[i]->informativePtnWgt = NULL;
            }
            if (pr->partitionData[i]->informativePtnScore != NULL) {
                rax_free(pr->partitionData[i]->informativePtnScore);
                pr->partitionData[i]->informativePtnScore = NULL;
            }
        }
        if (pllRemainderLowerBounds) {
            delete[] pllRemainderLowerBounds;
            pllRemainderLowerBounds = NULL;
        }
    }

    if (uppass_par) {
        rax_free(uppass_par);
        uppass_par = NULL;
    }
    if (branchNode) {
        rax_free(branchNode);
        branchNode = NULL;
    }
    if (depth) {
        rax_free(depth);
        depth = NULL;
    }
    if (distFromRmvBranch) {
        rax_free(distFromRmvBranch);
        distFromRmvBranch = NULL;
    }
    if (recalculate) {
        rax_free(recalculate);
        recalculate = NULL;
    }
    if (isUppassCopied) {
        rax_free(isUppassCopied);
        isUppassCopied = NULL;
    }
    if (inRadiusRange) {
        rax_free(inRadiusRange);
        inRadiusRange = NULL;
    }
    // cout << "End _pllFreeParsimonyDataStructuresUppass\n";
}

static void computeTraversalInfoParsimonyUppass(nodeptr p, int *ti,
                                                int *counter, int maxTips,
                                                int perSiteScores, bool first) {
    // if (first) {
    //     cout << "FIRST\n";
    // }
    // cout << "pNumber = " << p->number << '\n';
    assert(recalculate[p->number] == true);
    if (p->number <= maxTips) {
        ti[*counter] = p->number;
        if (first) {
            ti[*counter + 1] = 2 * maxTips - 1;
        } else {
            ti[*counter + 1] = p->back->number;
        }
        *counter = *counter + 4;
        return;
    }
    nodeptr q = p->next->back, r = p->next->next->back;
    // cout << "qNumber = " << q->number << '\n';
    // cout << "rNumber = " << r->number << '\n';

    if (recalculate[q->number]) {
        computeTraversalInfoParsimonyUppass(q, ti, counter, maxTips,
                                            perSiteScores, false);
    }

    if (recalculate[r->number]) {
        computeTraversalInfoParsimonyUppass(r, ti, counter, maxTips,
                                            perSiteScores, false);
    }
    ti[*counter] = p->number;
    ti[*counter + 1] = q->number;
    ti[*counter + 2] = r->number;
    if (first) {
        ti[*counter + 3] = 2 * maxTips - 1;
    } else {
        ti[*counter + 3] = p->back->number;
    }
    *counter = *counter + 4;
}
static void computeTraversalInfoParsimonyUppassFull(nodeptr p, int *ti,
                                                    int *counter, int maxTips,
                                                    int perSiteScores,
                                                    bool first) {
    // cout << "pNumber = " << p->number << '\n';
    if (p->number <= maxTips) {
        ti[*counter] = p->number;
        if (first) {
            ti[*counter + 1] = 2 * maxTips - 1;
        } else {
            ti[*counter + 1] = p->back->number;
        }
        *counter = *counter + 4;
        return;
    }
    nodeptr q = p->next->back, r = p->next->next->back;
    // cout << "qNumber = " << q->number << '\n';
    // cout << "rNumber = " << r->number << '\n';

    // if (!p->xPars)
    //     getxnodeLocal(p);

    computeTraversalInfoParsimonyUppassFull(q, ti, counter, maxTips,
                                            perSiteScores, false);

    computeTraversalInfoParsimonyUppassFull(r, ti, counter, maxTips,
                                            perSiteScores, false);
    ti[*counter] = p->number;
    ti[*counter + 1] = q->number;
    ti[*counter + 2] = r->number;
    if (first) {
        ti[*counter + 3] = 2 * maxTips - 1;
    } else {
        ti[*counter + 3] = p->back->number;
    }
    *counter = *counter + 4;
}

void assignAllLeavesDownpassToUppass(pllInstance *tr, partitionList *pr) {
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t states = pr->partitionData[model]->states,
               width = pr->partitionData[model]->parsimonyLength;
        int low = width * states;
        int high = width * states * ((size_t)tr->mxtips + 1);
        for (int i = low; i < high; ++i) {
            pr->partitionData[model]->parsVectUppass[i] =
                pr->partitionData[model]->parsVect[i];
        }
    }
}
void assignDownpassToUppass(partitionList *pr, size_t uNumber) {
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t states = pr->partitionData[model]->states,
               width = pr->partitionData[model]->parsimonyLength;
        int low = width * states * uNumber;
        int high = low + width * states;
        for (int i = low; i < high; ++i) {
            pr->partitionData[model]->parsVectUppass[i] =
                pr->partitionData[model]->parsVect[i];
        }
    }
}

void printUppass(pllInstance *tr, partitionList *pr, int u) {
    cout << "Uppass of Vertex: " << u << '\n';
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;
        switch (states) {
        default: {
//            parsimonyNumber *uStates[32];
            assert(states <= 32);
//            for (int k = 0; k < states; ++k) {
//                uStates[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppass[(width * states * u) + width * k]);
//            }
            parsimonyNumber *uStates = &(pr->partitionData[model]->parsVectUppass[table_index(width, states, u, 0, 0)]);

            cout << "width: " << width << '\n';
            for (int i = 0; i < width; i += INTS_PER_VECTOR) {

                for (int k = 0; k < states; ++k) {
                    if (k == 0) {
                        cout << "A: ";
                    } else if (k == 1) {
                        cout << "C: ";
                    } else if (k == 2) {
                        cout << "G: ";
                    } else if (k == 3) {
                        cout << "T: ";
                    }
//                    cout << bitset<8>(uStates[k][i]) << '\n';
                    cout << bitset<8>(uStates[table_index(width, states, 0, k, i)]) << "\n";
                }
            }
        }
        }
    }
}
void printAllUppass(pllInstance *tr, partitionList *pr) {
    for (size_t u = 1; u <= tr->mxtips * 2 - 2; ++u) {
        printUppass(tr, pr, u);
    }
}
#if (defined(__SSE3) || defined(__AVX))
/**
 * AVX version (Doesn't have normal version yet :<)
 */
template <typename Traits = SimdParsTraits>
void _newviewParsimonyIterativeFastUppass(pllInstance *tr, partitionList *pr,
                                          int perSiteScores) {
    if (pllCostMatrix) {
        assert(0);
    }

    INT_TYPE
    allOne = SET_ALL_BITS_ONE;

    int model, *ti = tr->ti, count = ti[0], index;

    for (index = 4; index < count; index += 4) {
        unsigned int totalScore = 0;

        size_t pNumber = (size_t)ti[index];
        if (pNumber <= tr->mxtips) {
            continue;
        }
        size_t qNumber = (size_t)ti[index + 1], rNumber = (size_t)ti[index + 2];

        if (perSiteScores) {
            if (qNumber <= tr->mxtips)
                resetPerSiteNodeScores(pr, qNumber);
            if (rNumber <= tr->mxtips)
                resetPerSiteNodeScores(pr, rNumber);
        }

        for (model = 0; model < pr->numberOfPartitions; ++model) {
            size_t k, states = pr->partitionData[model]->states,
                      width = pr->partitionData[model]->parsimonyLength;

            unsigned int *scoreInc =
                &(pr->partitionData[model]->scoreIncrease[width * pNumber]);

            auto idx = [width, states](int state_id, int col_id) -> int {
                return table_index(width, states, 0, state_id, col_id);
            };
            auto init = [width, states](std::size_t node_id) -> std::size_t {
                return table_index(width, states, node_id, 0, 0);
            };

            switch (states) {
            default: {
                assert(states <= 32);

                parsimonyNumber *left, *right, *cur;
                left = &(pr->partitionData[model]->parsVect[init(qNumber)]);
                right = &(pr->partitionData[model]->parsVect[init(rNumber)]);
                cur = &(pr->partitionData[model]->parsVect[init(pNumber)]);


                SimdParsVectProxy<Traits> _left(pr->partitionData[model]->parsVect + qNumber * width * states, states);
                SimdParsVectProxy<Traits> _right(pr->partitionData[model]->parsVect + rNumber * width * states, states);
                SimdParsVectProxy<Traits> _cur(pr->partitionData[model]->parsVect + pNumber * width * states, states);
                assert(init(qNumber) == qNumber * width * states);

                static_assert(Traits::size() == INTS_PER_VECTOR);

                for (std::size_t i = 0; i < width; i += Traits::size()) {
                    typename Traits::simd_type _v_N = Traits::zeros(), _l_A[32], _v_A[32];
//                    INT_TYPE s_r, s_l, v_N = SET_ALL_BITS_ZERO, l_A[32], v_A[32];

                    for (std::size_t j = 0; j < states; j++) {
//                        s_l = VECTOR_LOAD((CAST)(&left[idx(j, i)]));
//                        s_r = VECTOR_LOAD((CAST)(&right[idx(j, i)]));
//                        l_A[j] = VECTOR_BIT_AND(s_l, s_r);
//                        v_A[j] = VECTOR_BIT_OR(s_l, s_r);
//
//                        v_N = VECTOR_BIT_OR(v_N, l_A[j]);

                        auto _s_l = _left[i][j].load();
                        auto _s_r = _right[i][j].load();
                        _l_A[j] = _s_l & _s_r;
                        _v_A[j] = _s_l | _s_r;

                        _v_N |= _l_A[j];
//                        union {
//                            INT_TYPE v;
//                            std::uint32_t arr[8];
//                        } u;
//                        u.v = v_N;
//                        for (int k = 0; k < 8; k++) {
//                            assert(u.arr[k] == _v_N[k]);
//                        }
                    };

                    for (std::size_t j = 0; j < states; j++) {
                        _cur[i][j].store(_l_A[j] | andnot(_v_A[j], _v_N));
//                        VECTOR_STORE(
////                            (CAST)(&cur[j][i]),
//                                (CAST)(&cur[idx(j, i)]),
//                                VECTOR_BIT_OR(l_A[j], VECTOR_AND_NOT(v_N, v_A[j])));
                    }

                    _v_N = ~_v_N;
//                    v_N = VECTOR_AND_NOT(v_N, allOne);
//                    union {
//                        INT_TYPE v;
//                        std::uint32_t arr[8];
//                    } u;
//                    u.v = v_N;
//                    for (int k = 0; k < 8; k++) {
//                        assert(u.arr[k] == _v_N[k]);
//                    }

                    // totalScore += vectorPopcount(v_N);
                    alignas(PLL_BYTE_ALIGNMENT) unsigned int counts[Traits::template size<unsigned int>()];

                    SimdProxy<Traits>(counts).store(_v_N);

//                    VECTOR_STORE((CAST)counts, v_N);

                    for (std::size_t ptr = 0; ptr < Traits::template size<unsigned int>(); ++ptr) {
                        scoreInc[i + ptr] = __builtin_popcount(counts[ptr]);
                        totalScore += scoreInc[i + ptr];
                    }
//                    if (perSiteScores)
//                        storePerSiteNodeScores<Traits>(pr, model, v_N, i, pNumber);
                }
            }
            }
        }

        tr->parsimonyScore[pNumber] = totalScore + tr->parsimonyScore[rNumber] +
                                      tr->parsimonyScore[qNumber];
        if (perSiteScores)
            addPerSiteSubtreeScores(
                pr, pNumber, qNumber,
                rNumber); // Diep: add rNumber and qNumber to pNumber
    }
}
/*
 * AVX version
 */
unsigned int evaluateInsertParsimonyUppass(pllInstance *tr, partitionList *pr,
                                           int i1Number, int i2Number) {
    unsigned int score = scoreTwoSubtrees;

    INT_TYPE allOne = SET_ALL_BITS_ONE;
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;
        auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
            return table_index_divisible(width, states, 0, state_id, col_id);
        };
        auto init = [width, states](std::size_t node_id) -> std::size_t {
            return table_index_divisible(width, states, node_id, 0, 0);
        };

        switch (states) {
        default: {
//            parsimonyNumber *i1States[32], *i2States[32];

            assert(states <= 32);

//            for (k = 0; k < states; ++k) {
//                i1States[k] =
//                    &(pr->partitionData[model]
//                          ->branchVectUppass[(width * states * i1Number) +
//                                             width * k]);
//                i2States[k] =
//                    &(pr->partitionData[model]
//                          ->branchVectUppass[(width * states * i2Number) +
//                                             width * k]);
//            }
            parsimonyNumber *i1States, *i2States;
            i1States = &(pr->partitionData[model]->branchVectUppass[init(i1Number)]);
            i2States = &(pr->partitionData[model]->branchVectUppass[init(i2Number)]);

            // cout << "width: " << width << '\n';
            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                INT_TYPE t_N = SET_ALL_BITS_ZERO;

                for (int k = 0; k < states; ++k) {
                    INT_TYPE t_A =
//                        VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&i1States[k][i])),
//                                       VECTOR_LOAD((CAST)(&i2States[k][i])));
                        VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&i1States[idx(k, i)])),
                                       VECTOR_LOAD((CAST)(&i2States[idx(k, i)])));
                    t_N = VECTOR_BIT_OR(t_N, t_A);
                }

                t_N = VECTOR_AND_NOT(t_N, allOne);

                // score += vectorPopcount(t_N);

                unsigned int counts[INTS_PER_VECTOR]
                    __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                int ptr;

                VECTOR_STORE((CAST)counts, t_N);

                for (ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                    score += __builtin_popcount(counts[ptr]);
                    if (score > tr->bestParsimony) {
                        return score;
                    }
                }
                //                 if(sum >= bestScore)
                //                   return sum;
            }
        }
        }
    }
    return score;
}
/*
 * AVX version
 */
unsigned int evaluateInsertParsimonyUppassTBR(pllInstance *tr,
                                              partitionList *pr, nodeptr p,
                                              nodeptr q) {
    unsigned int score = scoreTwoSubtrees;
    size_t pNumber = p->number, p1Number = p->back->number, qNumber = q->number,
           q1Number = q->back->number;

    INT_TYPE allOne = SET_ALL_BITS_ONE;
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;
        auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
            return table_index(width, states, 0, state_id, col_id);
        };
        auto init = [width, states](std::size_t node_id) -> std::size_t {
            return table_index(width, states, node_id, 0, 0);
        };

        switch (states) {
        default: {
//            parsimonyNumber *pStates[32], *p1States[32], *qStates[32],
//                *q1States[32];

            assert(states <= 32);

//            for (k = 0; k < states; ++k) {
//                pStates[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * pNumber) +
//                                                width * k]);
//                p1States[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * p1Number) +
//                                                width * k]);
//                qStates[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * qNumber) +
//                                                width * k]);
//                q1States[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * q1Number) +
//                                                width * k]);
//            }
            parsimonyNumber *pStates, *p1States, *qStates, *q1States;
            pStates = &(pr->partitionData[model]->parsVectUppassLocal[init(pNumber)]);
            p1States = &(pr->partitionData[model]->parsVectUppassLocal[init(p1Number)]);
            qStates = &(pr->partitionData[model]->parsVectUppassLocal[init(qNumber)]);
            q1States = &(pr->partitionData[model]->parsVectUppassLocal[init(q1Number)]);

            // cout << "width: " << width << '\n';
            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                INT_TYPE t_N = SET_ALL_BITS_ZERO;

                for (int k = 0; k < states; ++k) {
                    INT_TYPE t_A = VECTOR_BIT_AND(
//                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&pStates[k][i])),
//                                      VECTOR_LOAD((CAST)(&p1States[k][i]))),
                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&pStates[idx(k, i)])),
                                      VECTOR_LOAD((CAST)(&p1States[idx(k, i)]))),
//                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&qStates[k][i])),
//                                      VECTOR_LOAD((CAST)(&q1States[k][i]))));
                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&qStates[idx(k, i)])),
                                      VECTOR_LOAD((CAST)(&q1States[idx(k, i)]))));
                    t_N = VECTOR_BIT_OR(t_N, t_A);
                }

                t_N = VECTOR_AND_NOT(t_N, allOne);

                // score += vectorPopcount(t_N);

                unsigned int counts[INTS_PER_VECTOR]
                    __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                int ptr;

                VECTOR_STORE((CAST)counts, t_N);

                for (ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                    score += __builtin_popcount(counts[ptr]);
                    if (score > tr->bestParsimony) {
                        return score;
                    }
                }
                //                 if(sum >= bestScore)
                //                   return sum;
            }
        }
        }
    }
    return score;
}
/*
 * AVX version
 */
unsigned int evaluateInsertParsimonyUppassSPR(pllInstance *tr,
                                              partitionList *pr, nodeptr p,
                                              nodeptr u) {
    unsigned int score = scoreTwoSubtrees;
    size_t pNumber = p->number, uNumber = u->number, vNumber = u->back->number;

    INT_TYPE allOne = SET_ALL_BITS_ONE;
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;
        auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
            return table_index(width, states, 0, state_id, col_id);
        };
        auto init = [width, states](std::size_t node_id) -> std::size_t {
            return table_index(width, states, node_id, 0, 0);
        };

        switch (states) {
        default: {
//            parsimonyNumber *pStates[32], *uStates[32], *vStates[32];

            assert(states <= 32);

//            for (k = 0; k < states; ++k) {
//                pStates[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * pNumber) +
//                                                width * k]);
//                uStates[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * uNumber) +
//                                                width * k]);
//                vStates[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * vNumber) +
//                                                width * k]);
//            }
            parsimonyNumber *pStates, *uStates, *vStates;
            pStates = &(pr->partitionData[model]->parsVectUppassLocal[init(pNumber)]);
            uStates = &(pr->partitionData[model]->parsVectUppassLocal[init(uNumber)]);
            vStates = &(pr->partitionData[model]->parsVectUppassLocal[init(vNumber)]);

            // cout << "width: " << width << '\n';
            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                INT_TYPE t_N = SET_ALL_BITS_ZERO;

                for (int k = 0; k < states; ++k) {
                    INT_TYPE t_A = VECTOR_BIT_AND(
//                        VECTOR_LOAD((CAST)(&pStates[k][i])),
                        VECTOR_LOAD((CAST)(&pStates[idx(k, i)])),
//                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&uStates[k][i])),
//                                      VECTOR_LOAD((CAST)(&vStates[k][i]))));
                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&uStates[idx(k, i)])),
                                      VECTOR_LOAD((CAST)(&vStates[idx(k, i)]))));
                    t_N = VECTOR_BIT_OR(t_N, t_A);
                }

                t_N = VECTOR_AND_NOT(t_N, allOne);

                // score += vectorPopcount(t_N);
                unsigned int counts[INTS_PER_VECTOR]
                    __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                int ptr;

                VECTOR_STORE((CAST)counts, t_N);

                for (ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                    score += __builtin_popcount(counts[ptr]);
                    if (score > tr->bestParsimony) {
                        return score;
                    }
                }
                //                 if(sum >= bestScore)
                //                   return sum;
            }
        }
        }
    }
    return score;
}

/*
 * AVX version
 */
void uppassStatesIterativeCalculate(pllInstance *tr, partitionList *pr) {
    // auto t_start = std::chrono::high_resolution_clock::now();
    if (pllCostMatrix) {
        assert(0);
    }
    INT_TYPE
    allOne = SET_ALL_BITS_ONE;
    int model, *ti = tr->ti, count = ti[0], index;
    // assert(count % 4 == 0);
    for (index = count - 4; index > 0; index -= 4) {
        unsigned int totalScore = 0;
        size_t uNumber = (size_t)ti[index];
        if (uNumber <= tr->mxtips) {
            size_t pNumber = (size_t)ti[index + 1];
            for (model = 0; model < pr->numberOfPartitions; ++model) {
                size_t k, states = pr->partitionData[model]->states,
                          width = pr->partitionData[model]->parsimonyLength;
                auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
                    return table_index(width, states, 0, state_id, col_id);
                };
                auto init = [width, states](std::size_t node_id) -> std::size_t {
                    return table_index(width, states, node_id, 0, 0);
                };

                unsigned int i;

                switch (states) {
                /* TODO: Add case 2, 4, 10 */
                default: {
                    /**
                     * u:    downpass state of current leaf
                     * p:    uppass state of parent node
                     */
                    assert(states <= 32);
//                    parsimonyNumber *u[32], *p[32], *uUppass[32];

//                    for (k = 0; k < states; ++k) {
//                        u[k] = &(pr->partitionData[model]
//                                     ->parsVect[(width * states * uNumber) +
//                                                width * k]);
//                        uUppass[k] =
//                            &(pr->partitionData[model]
//                                  ->parsVectUppass[(width * states * uNumber) +
//                                                   width * k]);
//                        p[k] =
//                            &(pr->partitionData[model]
//                                  ->parsVectUppass[(width * states * pNumber) +
//                                                   width * k]);
//                    }
                    parsimonyNumber *u, *p, *uUppass;
                    u = &(pr->partitionData[model]->parsVect[init(uNumber)]);
                    uUppass = &(pr->partitionData[model]->parsVectUppass[init(uNumber)]);
                    p = &(pr->partitionData[model]->parsVectUppass[init(pNumber)]);

                    INT_TYPE x, u_k[32], p_k[32];
                    for (i = 0; i < width; i += INTS_PER_VECTOR) {
                        x = SET_ALL_BITS_ZERO;
                        for (k = 0; k < states; ++k) {
//                            u_k[k] = VECTOR_LOAD((CAST)(&u[k][i]));
                            u_k[k] = VECTOR_LOAD((CAST)(&u[idx(k, i)]));
//                            p_k[k] = VECTOR_LOAD((CAST)(&p[k][i]));
                            p_k[k] = VECTOR_LOAD((CAST)(&p[idx(k, i)]));
                            x = VECTOR_BIT_OR(
                                x,
                                VECTOR_BIT_AND(
                                    VECTOR_AND_NOT(
                                        VECTOR_BIT_AND(u_k[k], p_k[k]), allOne),
                                    p_k[k]));
                            // x |= (~(u[k][i] & p[k][i]) & p[k][i]);
                        }
                        x = VECTOR_AND_NOT(x, allOne);
                        for (k = 0; k < states; ++k) {
                            u_k[k] = VECTOR_BIT_XOR(
                                u_k[k], VECTOR_BIT_AND(
                                            VECTOR_BIT_XOR(u_k[k], p_k[k]), x));
//                            VECTOR_STORE((CAST)(&uUppass[k][i]), u_k[k]);
                            VECTOR_STORE((CAST)(&uUppass[idx(k, i)]), u_k[k]);
                            // u[k][i] ^= ((u[k][i] ^ p[k][i]) & x);
                        }
                    }
                }
                }
            }
        } else {
            size_t v1Number = (size_t)ti[index + 1],
                   v2Number = (size_t)ti[index + 2],
                   pNumber = (size_t)ti[index + 3];

            for (model = 0; model < pr->numberOfPartitions; ++model) {
                size_t k, states = pr->partitionData[model]->states,
                          width = pr->partitionData[model]->parsimonyLength;
                auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
                    return table_index(width, states, 0, state_id, col_id);
                };
                auto init = [width, states](std::size_t node_id) -> std::size_t {
                    return table_index(width, states, node_id, 0, 0);
                };

                unsigned int i;

                switch (states) {
                default: {
                    /**
                     * u:    downpass state of current node
                     * p:    uppass state of parent node
                     * v_1:  downpass state of children 1 of u
                     * v_2:  downpass state of children 2 of u
                     */
                    assert(states <= 32);
//                    parsimonyNumber *u[32], *v_1[32], *v_2[32], *p[32],
//                        *uUppass[32];
//
//                    for (k = 0; k < states; ++k) {
//                        u[k] = &(pr->partitionData[model]
//                                     ->parsVect[(width * states * uNumber) +
//                                                width * k]);
//                        v_1[k] = &(pr->partitionData[model]
//                                       ->parsVect[(width * states * v1Number) +
//                                                  width * k]);
//                        v_2[k] = &(pr->partitionData[model]
//                                       ->parsVect[(width * states * v2Number) +
//                                                  width * k]);
//                        uUppass[k] =
//                            &(pr->partitionData[model]
//                                  ->parsVectUppass[(width * states * uNumber) +
//                                                   width * k]);
//                        p[k] =
//                            &(pr->partitionData[model]
//                                  ->parsVectUppass[(width * states * pNumber) +
//                                                   width * k]);
//                    }
                    parsimonyNumber *u, *v_1, *v_2, *p, *uUppass;
                    u = &(pr->partitionData[model]->parsVect[init(uNumber)]);
                    v_1 = &(pr->partitionData[model]->parsVect[init(v1Number)]);
                    v_2 = &(pr->partitionData[model]->parsVect[init(v2Number)]);
                    uUppass = &(pr->partitionData[model]->parsVectUppass[init(uNumber)]);
                    p = &(pr->partitionData[model]->parsVectUppass[init(pNumber)]);

                    INT_TYPE x = SET_ALL_BITS_ZERO, y = SET_ALL_BITS_ZERO,
                             u_k[32], p_k[32], v_1k[32], v_2k[32], u_up;
                    for (i = 0; i < width; i += INTS_PER_VECTOR) {
                        x = SET_ALL_BITS_ZERO, y = SET_ALL_BITS_ZERO;
                        for (k = 0; k < states; ++k) {
//                            u_k[k] = VECTOR_LOAD((CAST)(&u[k][i]));
                            u_k[k] = VECTOR_LOAD((CAST)(&u[idx(k, i)]));
//                            p_k[k] = VECTOR_LOAD((CAST)(&p[k][i]));
                            p_k[k] = VECTOR_LOAD((CAST)(&p[idx(k, i)]));
//                            v_1k[k] = VECTOR_LOAD((CAST)(&v_1[k][i]));
                            v_1k[k] = VECTOR_LOAD((CAST)(&v_1[idx(k, i)]));
//                            v_2k[k] = VECTOR_LOAD((CAST)(&v_2[k][i]));
                            v_2k[k] = VECTOR_LOAD((CAST)(&v_2[idx(k, i)]));
                            x = VECTOR_BIT_OR(
                                x,
                                VECTOR_BIT_AND(
                                    VECTOR_AND_NOT(
                                        VECTOR_BIT_AND(u_k[k], p_k[k]), allOne),
                                    p_k[k]));
                            y = VECTOR_BIT_OR(y,
                                              VECTOR_BIT_AND(v_1k[k], v_2k[k]));

                            // x |= (~(u[k][i] & p[k][i]) & p[k][i]);
                            // y |= (v_1[k][i] & v_2[k][i]);
                        }
                        for (k = 0; k < states; ++k) {
                            // is it possible to set u_up = u_k?
                            u_up = u_k[k];

                            u_up = VECTOR_BIT_XOR(
                                u_up,
                                VECTOR_BIT_AND(VECTOR_BIT_XOR(u_up, p_k[k]),
                                               VECTOR_AND_NOT(x, allOne)));
                            u_up = VECTOR_BIT_XOR(
                                u_up,
                                VECTOR_BIT_AND(
                                    VECTOR_BIT_XOR(u_up,
                                                   VECTOR_BIT_OR(u_up, p_k[k])),
                                    VECTOR_BIT_AND(x,
                                                   VECTOR_AND_NOT(y, allOne))));
                            u_up = VECTOR_BIT_XOR(
                                u_up,
                                VECTOR_BIT_AND(
                                    VECTOR_BIT_XOR(
                                        u_up,
                                        VECTOR_BIT_OR(
                                            u_up, VECTOR_BIT_AND(
                                                      p_k[k],
                                                      VECTOR_BIT_OR(v_1k[k],
                                                                    v_2k[k])))),
                                    VECTOR_BIT_AND(x, y)));
//                            VECTOR_STORE((CAST)(&uUppass[k][i]), u_up);
                            VECTOR_STORE((CAST)(&uUppass[idx(k, i)]), u_up);
                            // uUppass[k][i] = u[k][i];
                            // uUppass[k][i] ^= ((uUppass[k][i] ^ p[k][i]) &
                            // (~x)); uUppass[k][i] ^=
                            //     ((uUppass[k][i] ^ (uUppass[k][i] | p[k][i]))
                            //     &
                            //      (x & (~y)));
                            // uUppass[k][i] ^=
                            //     ((uUppass[k][i] ^
                            //       (uUppass[k][i] |
                            //        (p[k][i] & (v_1[k][i] | v_2[k][i])))) &
                            //      (x & y));
                        }
                    }
                }
                }
            }
        }
    }
    // auto t_end = std::chrono::high_resolution_clock::now();
    // double elapsed_time_ms =
    //     std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // total_time_uppass += elapsed_time_ms;
    // cout << "Done uppassStatesIterativeCalculate\n";
}

/*
 * AVX version
 */
template <typename Traits = SimdParsTraits>
unsigned int _evaluateParsimonyIterativeFastUppass(pllInstance *tr,
                                                   partitionList *pr,
                                                   int perSiteScores) {
    if (pllCostMatrix) {
        assert(0);
    }
    size_t pNumber = (size_t)tr->ti[1], qNumber = (size_t)tr->ti[2];

    INT_TYPE allOne = SET_ALL_BITS_ONE;

    size_t temNumber = 2 * (size_t)tr->mxtips - 1;
    int model;

    unsigned int sum;

    if (tr->ti[0] > 4)
        _newviewParsimonyIterativeFastUppass(tr, pr, perSiteScores);

    sum = tr->parsimonyScore[pNumber] + tr->parsimonyScore[qNumber];

    if (perSiteScores) {
        resetPerSiteNodeScores(pr, tr->start->number);
        addPerSiteSubtreeScores(pr, tr->start->number, pNumber, qNumber);
    }
    for (model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;
        auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
            return table_index_divisible(width, states, 0, state_id, col_id);
        };
        auto init = [width, states](std::size_t node_id) -> std::size_t {
            return table_index_divisible(width, states, node_id, 0, 0);
        };

        unsigned int *scoreInc =
            &(pr->partitionData[model]
                  ->scoreIncrease[width * (2 * tr->mxtips - 1)]);
        switch (states) {
        case 2: {
//            parsimonyNumber *left[2], *right[2], *tem[2];

//            for (k = 0; k < 2; ++k) {
//                left[k] = &(pr->partitionData[model]
//                                ->parsVect[(width * 2 * qNumber) + width * k]);
//                right[k] = &(pr->partitionData[model]
//                                 ->parsVect[(width * 2 * pNumber) + width * k]);
//                tem[k] = &(
//                    pr->partitionData[model]
//                        ->parsVectUppass[(width * 2 * temNumber) + width * k]);
//            }
            parsimonyNumber *left, *right, *tem;
            left = &(pr->partitionData[model]->parsVect[init(qNumber)]);
            right = &(pr->partitionData[model]->parsVect[init(pNumber)]);
            tem = &(pr->partitionData[model]->parsVectUppass[init(temNumber)]);

            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                INT_TYPE t_A, t_C, o_A, o_C, t_N, s_l, s_r;
//                s_l = VECTOR_LOAD((CAST)(&left[0][i]));
                s_l = VECTOR_LOAD((CAST)(&left[idx(0, i)]));
//                s_r = VECTOR_LOAD((CAST)(&right[0][i]));
                s_r = VECTOR_LOAD((CAST)(&right[idx(0, i)]));
                t_A = VECTOR_BIT_AND(s_l, s_r);
                o_A = VECTOR_BIT_OR(s_l, s_r);

//                s_l = VECTOR_LOAD((CAST)(&left[1][i]));
                s_l = VECTOR_LOAD((CAST)(&left[idx(1, i)]));
//                s_r = VECTOR_LOAD((CAST)(&right[1][i]));
                s_r = VECTOR_LOAD((CAST)(&right[idx(1, i)]));
                t_C = VECTOR_BIT_AND(s_l, s_r);
                o_C = VECTOR_BIT_OR(s_l, s_r);

                t_N = VECTOR_BIT_OR(t_A, t_C);
                t_N = VECTOR_AND_NOT(t_N, allOne);

//                VECTOR_STORE((CAST)(&tem[0][i]),
                VECTOR_STORE((CAST)(&tem[idx(0, i)]),
                             VECTOR_BIT_OR(t_A, VECTOR_BIT_AND(t_N, o_A)));
//                VECTOR_STORE((CAST)(&tem[1][i]),
                VECTOR_STORE((CAST)(&tem[idx(1, i)]),
                             VECTOR_BIT_OR(t_C, VECTOR_BIT_AND(t_N, o_C)));

                // sum += vectorPopcount(t_N);
                unsigned int counts[INTS_PER_VECTOR]
                    __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                int ptr;

                VECTOR_STORE((CAST)counts, t_N);

                for (ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                    scoreInc[i + ptr] = __builtin_popcount(counts[ptr]);
                    sum += scoreInc[i + ptr];
                }
//                if (perSiteScores)
//                    storePerSiteNodeScores(pr, model, t_N, i, pNumber);
                //                 if(sum >= bestScore)
                //                   return sum;
            }
        } break;
        case 4: {
//            parsimonyNumber *left[4], *right[4], *tem[4];

//            for (k = 0; k < 4; ++k) {
//                left[k] = &(pr->partitionData[model]
//                                ->parsVect[(width * 4 * qNumber) + width * k]);
//                right[k] = &(pr->partitionData[model]
//                                 ->parsVect[(width * 4 * pNumber) + width * k]);
//                tem[k] = &(
//                    pr->partitionData[model]
//                        ->parsVectUppass[(width * 4 * temNumber) + width * k]);
//            }
            parsimonyNumber *left, *right, *tem;
            left = &(pr->partitionData[model]->parsVect[init(qNumber)]);
            right = &(pr->partitionData[model]->parsVect[init(pNumber)]);
            tem = &(pr->partitionData[model]->parsVectUppass[init(temNumber)]);

            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                INT_TYPE t_A, t_C, t_G, t_T, o_A, o_C, o_G, o_T, t_N, s_l, s_r;
//                s_l = VECTOR_LOAD((CAST)(&left[0][i]));
                s_l = VECTOR_LOAD((CAST)(&left[idx(0, i)]));
//                s_r = VECTOR_LOAD((CAST)(&right[0][i]));
                s_r = VECTOR_LOAD((CAST)(&right[idx(0, i)]));
                t_A = VECTOR_BIT_AND(s_l, s_r);
                o_A = VECTOR_BIT_OR(s_l, s_r);

//                s_l = VECTOR_LOAD((CAST)(&left[1][i]));
                s_l = VECTOR_LOAD((CAST)(&left[idx(1, i)]));
//                s_r = VECTOR_LOAD((CAST)(&right[1][i]));
                s_r = VECTOR_LOAD((CAST)(&right[idx(1, i)]));
                t_C = VECTOR_BIT_AND(s_l, s_r);
                o_C = VECTOR_BIT_OR(s_l, s_r);

//                s_l = VECTOR_LOAD((CAST)(&left[2][i]));
                s_l = VECTOR_LOAD((CAST)(&left[idx(2, i)]));
//                s_r = VECTOR_LOAD((CAST)(&right[2][i]));
                s_r = VECTOR_LOAD((CAST)(&right[idx(2, i)]));
                t_G = VECTOR_BIT_AND(s_l, s_r);
                o_G = VECTOR_BIT_OR(s_l, s_r);

//                s_l = VECTOR_LOAD((CAST)(&left[3][i]));
                s_l = VECTOR_LOAD((CAST)(&left[idx(3, i)]));
//                s_r = VECTOR_LOAD((CAST)(&right[3][i]));
                s_r = VECTOR_LOAD((CAST)(&right[idx(3, i)]));
                t_T = VECTOR_BIT_AND(s_l, s_r);
                o_T = VECTOR_BIT_OR(s_l, s_r);

                t_N = VECTOR_BIT_OR(VECTOR_BIT_OR(t_A, t_C),
                                    VECTOR_BIT_OR(t_G, t_T));
                t_N = VECTOR_AND_NOT(t_N, allOne);

//                VECTOR_STORE((CAST)(&tem[0][i]),
                VECTOR_STORE((CAST)(&tem[idx(0, i)]),
                             VECTOR_BIT_OR(t_A, VECTOR_BIT_AND(t_N, o_A)));
//                VECTOR_STORE((CAST)(&tem[1][i]),
                VECTOR_STORE((CAST)(&tem[idx(1, i)]),
                             VECTOR_BIT_OR(t_C, VECTOR_BIT_AND(t_N, o_C)));
//                VECTOR_STORE((CAST)(&tem[2][i]),
                VECTOR_STORE((CAST)(&tem[idx(2, i)]),
                             VECTOR_BIT_OR(t_G, VECTOR_BIT_AND(t_N, o_G)));
//                VECTOR_STORE((CAST)(&tem[3][i]),
                VECTOR_STORE((CAST)(&tem[idx(3, i)]),
                             VECTOR_BIT_OR(t_T, VECTOR_BIT_AND(t_N, o_T)));
                // sum += vectorPopcount(t_N);
                unsigned int counts[INTS_PER_VECTOR]
                    __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                int ptr;

                VECTOR_STORE((CAST)counts, t_N);

                for (ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                    scoreInc[i + ptr] = __builtin_popcount(counts[ptr]);
                    sum += scoreInc[i + ptr];
                }

//                if (perSiteScores)
//                    storePerSiteNodeScores(pr, model, t_N, i, pNumber);
                //                 if(sum >= bestScore)
                //                   return sum;
            }
        } break;
        case 20: {
//            parsimonyNumber *left[20], *right[20], *tem[20];
//            for (k = 0; k < 20; ++k) {
//                left[k] = &(pr->partitionData[model]
//                                ->parsVect[(width * 20 * qNumber) + width * k]);
//                right[k] =
//                    &(pr->partitionData[model]
//                          ->parsVect[(width * 20 * pNumber) + width * k]);
//                tem[k] = &(
//                    pr->partitionData[model]
//                        ->parsVectUppass[(width * 20 * temNumber) + width * k]);
//            }
            parsimonyNumber *left, *right, *tem;
            left = &(pr->partitionData[model]->parsVect[init(qNumber)]);
            right = &(pr->partitionData[model]->parsVect[init(pNumber)]);
            tem = &(pr->partitionData[model]->parsVectUppass[init(temNumber)]);

            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                INT_TYPE s_l, s_r, t_N = SET_ALL_BITS_ZERO, t_A[20], o_A[20];

                for (k = 0; k < 20; ++k) {
//                    s_l = VECTOR_LOAD((CAST)(&left[k][i]));
                    s_l = VECTOR_LOAD((CAST)(&left[idx(k, i)]));
//                    s_r = VECTOR_LOAD((CAST)(&right[k][i]));
                    s_r = VECTOR_LOAD((CAST)(&right[idx(k, i)]));
                    t_A[k] = VECTOR_BIT_AND(s_l, s_r);
                    o_A[k] = VECTOR_BIT_OR(s_l, s_r);
                    t_N = VECTOR_BIT_OR(t_N, t_A[k]);
                }

                t_N = VECTOR_AND_NOT(t_N, allOne);

                for (k = 0; k < 20; ++k) {
                    VECTOR_STORE(
//                        (CAST)(&tem[k][i]),
                        (CAST)(&tem[idx(k, i)]),
                        VECTOR_BIT_OR(t_A[k], VECTOR_BIT_AND(t_N, o_A[k])));
                }
                // sum += vectorPopcount(t_N);
                unsigned int counts[INTS_PER_VECTOR]
                    __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                int ptr;

                VECTOR_STORE((CAST)counts, t_N);

                for (ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                    scoreInc[i + ptr] = __builtin_popcount(counts[ptr]);
                    sum += scoreInc[i + ptr];
                }
//                if (perSiteScores)
//                    storePerSiteNodeScores(pr, model, t_N, i, pNumber);
                //                  if(sum >= bestScore)
                //                    return sum;
            }
        } break;
        default: {
//            parsimonyNumber *left[32], *right[32], *tem[32];
            assert(states <= 32);
//            for (k = 0; k < states; ++k) {
//                left[k] =
//                    &(pr->partitionData[model]
//                          ->parsVect[(width * states * qNumber) + width * k]);
//                right[k] =
//                    &(pr->partitionData[model]
//                          ->parsVect[(width * states * pNumber) + width * k]);
//                tem[k] = &(pr->partitionData[model]
//                               ->parsVectUppass[(width * states * temNumber) +
//                                                width * k]);
//            }
            parsimonyNumber *left, *right, *tem;
            left = &(pr->partitionData[model]->parsVect[init(qNumber)]);
            right = &(pr->partitionData[model]->parsVect[init(pNumber)]);
            tem = &(pr->partitionData[model]->parsVectUppass[init(temNumber)]);

            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                INT_TYPE s_l, s_r, t_N = SET_ALL_BITS_ZERO, t_A[32], o_A[32];

                for (k = 0; k < states; ++k) {
//                    s_l = VECTOR_LOAD((CAST)(&left[k][i]));
                    s_l = VECTOR_LOAD((CAST)(&left[idx(k, i)]));
//                    s_r = VECTOR_LOAD((CAST)(&right[k][i]));
                    s_r = VECTOR_LOAD((CAST)(&right[idx(k, i)]));
                    t_A[k] = VECTOR_BIT_AND(s_l, s_r);
                    o_A[k] = VECTOR_BIT_OR(s_l, s_r);
                    t_N = VECTOR_BIT_OR(t_N, t_A[k]);
                }

                t_N = VECTOR_AND_NOT(t_N, allOne);

                for (k = 0; k < states; ++k) {
                    VECTOR_STORE(
//                        (CAST)(&tem[k][i]),
                        (CAST)(&tem[idx(k, i)]),
                        VECTOR_BIT_OR(t_A[k], VECTOR_BIT_AND(t_N, o_A[k])));
                }
                // sum += vectorPopcount(t_N);
                unsigned int counts[INTS_PER_VECTOR]
                    __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                int ptr;

                VECTOR_STORE((CAST)counts, t_N);

                for (ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                    scoreInc[i + ptr] = __builtin_popcount(counts[ptr]);
                    sum += scoreInc[i + ptr];
                }
//                if (perSiteScores)
//                    storePerSiteNodeScores<Traits>(pr, model, t_N, i, pNumber);
                //                  if(sum >= bestScore)
                //                    return sum;
            }
        }
        }
    }

    // cout << "Sum = " << sum << '\n';
    uppassStatesIterativeCalculate(tr, pr);
    return sum;
}

void traversePrepareInsertBranches(partitionList *pr, nodeptr u, int maxTips,
                                   int &count) {
    size_t uNumber = u->number, u1Number = u->back->number;
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;
        auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
            return table_index_divisible(width, states, 0, state_id, col_id);
        };
        auto init = [width, states](std::size_t node_id) -> std::size_t {
            return table_index_divisible(width, states, node_id, 0, 0);
        };

        switch (states) {
        default: {
//            parsimonyNumber *uStates[32], *u1States[32], *branchStates[32];

            assert(states <= 32);

//            for (k = 0; k < states; ++k) {
//                uStates[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * uNumber) +
//                                                width * k]);
//                u1States[k] =
//                    &(pr->partitionData[model]
//                          ->parsVectUppassLocal[(width * states * u1Number) +
//                                                width * k]);
//                branchStates[k] =
//                    &(pr->partitionData[model]
//                          ->branchVectUppass[(width * states * count) +
//                                             width * k]);
//            }
            parsimonyNumber *uStates, *u1States, *branchStates;
            uStates = &(pr->partitionData[model]->parsVectUppassLocal[init(uNumber)]);
            u1States = &(pr->partitionData[model]->parsVectUppassLocal[init(u1Number)]);
            branchStates = &(pr->partitionData[model]->branchVectUppass[init(count)]);


            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                for (int k = 0; k < states; ++k) {
                    VECTOR_STORE(
//                        (CAST)&branchStates[k][i],
                        (CAST)&branchStates[idx(k, i)],
//                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&uStates[k][i])),
//                                      VECTOR_LOAD((CAST)(&u1States[k][i]))));
                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&uStates[idx(k, i)])),
                                      VECTOR_LOAD((CAST)(&u1States[idx(k, i)]))));
                }
            }
        }
        }
    }
    branchNode[count] = u;
    count++;
    if (u->number > maxTips) {
        traversePrepareInsertBranches(pr, u->next->back, maxTips, count);
        traversePrepareInsertBranches(pr, u->next->next->back, maxTips, count);
    }
}

void traversePrepareInsertBranches(partitionList *pr, nodeptr u, int maxtrav,
                                   int dist, int maxTips, int &count) {
    size_t uNumber = u->number, u1Number = u->back->number;
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;
        auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
            return table_index_divisible(width, states, 0, state_id, col_id);
        };
        auto init = [width, states](std::size_t node_id) -> std::size_t {
            return table_index_divisible(width, states, node_id, 0, 0);
        };

        switch (states) {
        default: {
//            parsimonyNumber *uStates[32], *u1States[32], *branchStates[32];

            assert(states <= 32);

//            for (k = 0; k < states; ++k) {
//                if (isUppassCopied[uNumber]) {
//                    uStates[k] =
//                        &(pr->partitionData[model]
//                              ->parsVectUppassLocal[(width * states * uNumber) +
//                                                    width * k]);
//                } else {
//                    uStates[k] =
//                        &(pr->partitionData[model]
//                              ->parsVectUppass[(width * states * uNumber) +
//                                               width * k]);
//                }
//                if (isUppassCopied[u1Number]) {
//                    u1States[k] = &(
//                        pr->partitionData[model]
//                            ->parsVectUppassLocal[(width * states * u1Number) +
//                                                  width * k]);
//                } else {
//                    u1States[k] =
//                        &(pr->partitionData[model]
//                              ->parsVectUppass[(width * states * u1Number) +
//                                               width * k]);
//                }
//                branchStates[k] =
//                    &(pr->partitionData[model]
//                          ->branchVectUppass[(width * states * count) +
//                                             width * k]);
//            }
            parsimonyNumber *uStates, *u1States, *branchStates;
            if (isUppassCopied[uNumber]) {
                uStates = &(pr->partitionData[model]->parsVectUppassLocal[init(uNumber)]);
            } else {
                uStates = &(pr->partitionData[model]->parsVectUppass[init(uNumber)]);
            }
            if (isUppassCopied[u1Number]) {
                u1States = &(pr->partitionData[model]->parsVectUppassLocal[init(u1Number)]);
            } else {
                u1States = &(pr->partitionData[model]->parsVectUppass[init(u1Number)]);
            }
            branchStates = &(pr->partitionData[model]->branchVectUppass[init(count)]);

            for (i = 0; i < width; i += INTS_PER_VECTOR) {
                for (int k = 0; k < states; ++k) {
                    VECTOR_STORE(
//                        (CAST)&branchStates[k][i],
                        (CAST)&branchStates[idx(k, i)],
//                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&uStates[k][i])),
//                                      VECTOR_LOAD((CAST)(&u1States[k][i]))));
                        VECTOR_BIT_OR(VECTOR_LOAD((CAST)(&uStates[idx(k, i)])),
                                      VECTOR_LOAD((CAST)(&u1States[idx(k, i)]))));
                }
            }
        }
        }
    }
    branchNode[count] = u;
    distFromRmvBranch[count] = dist;
    count++;
    if (u->number > maxTips && maxtrav >= 1) {
        traversePrepareInsertBranches(pr, u->next->back, maxtrav - 1, dist + 1,
                                      maxTips, count);
        traversePrepareInsertBranches(pr, u->next->next->back, maxtrav - 1,
                                      dist + 1, maxTips, count);
    }
}
#else
unsigned int evaluateInsertParsimonyUppassSPR(pllInstance *tr,
                                              partitionList *pr, nodeptr p,
                                              nodeptr u) {
    unsigned int score = scoreTwoSubtrees;
    size_t pNumber = p->number, uNumber = u->number, vNumber = u->back->number;

    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;
        switch (states) {
        default: {
            parsimonyNumber t_A, t_N, *pStates[32], *uStates[32], *vStates[32];

            assert(states <= 32);

            for (int k = 0; k < states; ++k) {
                pStates[k] = &(pr->partitionData[model]
                                   ->parsVectUppass[(width * states * pNumber) +
                                                    width * k]);
                uStates[k] = &(pr->partitionData[model]
                                   ->parsVectUppass[(width * states * uNumber) +
                                                    width * k]);
                vStates[k] = &(pr->partitionData[model]
                                   ->parsVectUppass[(width * states * vNumber) +
                                                    width * k]);
            }

            // cout << "width: " << width << '\n';
            for (int i = 0; i < width; ++i) {
                t_N = 0;

                for (int k = 0; k < states; ++k) {
                    t_A = pStates[k][i] & (uStates[k][i] | vStates[k][i]);
                    t_N = t_N | t_A;
                }

                t_N = ~t_N;

                score += ((unsigned int)__builtin_popcount(t_N));

                if (score > tr->bestParsimony) {
                    return score;
                }
                //                 if(sum >= bestScore)
                //                   return sum;
            }
        }
        }
    }
    return score;
}
void uppassStatesIterativeCalculate(pllInstance *tr, partitionList *pr) {
    if (pllCostMatrix) {
        assert(0);
    }
    int model, *ti = tr->ti, count = ti[0], index;
    // assert(count % 4 == 0);
    for (index = count - 4; index > 0; index -= 4) {
        size_t uNumber = (size_t)ti[index];
        if (uNumber <= tr->mxtips) {
            size_t pNumber = (size_t)ti[index + 1];
            for (model = 0; model < pr->numberOfPartitions; ++model) {
                size_t k, states = pr->partitionData[model]->states,
                          width = pr->partitionData[model]->parsimonyLength;

                unsigned int i;

                switch (states) {
                default: {
                    /**
                     * u:    downpass state of current leaf
                     * p:    uppass state of parent node
                     */
                    assert(states <= 32);
                    parsimonyNumber *u[32], *p[32], *uUppass[32];

                    for (k = 0; k < states; ++k) {
                        u[k] = &(pr->partitionData[model]
                                     ->parsVect[(width * states * uNumber) +
                                                width * k]);
                        uUppass[k] =
                            &(pr->partitionData[model]
                                  ->parsVectUppass[(width * states * uNumber) +
                                                   width * k]);
                        p[k] =
                            &(pr->partitionData[model]
                                  ->parsVectUppass[(width * states * pNumber) +
                                                   width * k]);
                    }

                    for (i = 0; i < width; ++i) {
                        parsimonyNumber x = 0;
                        for (k = 0; k < states; ++k) {
                            x |= (~(u[k][i] & p[k][i]) & p[k][i]);
                        }
                        x = ~x;
                        for (k = 0; k < states; ++k) {
                            uUppass[k][i] = u[k][i];
                            uUppass[k][i] ^= ((u[k][i] ^ p[k][i]) & x);
                        }
                    }
                }
                }
            }
        } else {
            size_t v1Number = (size_t)ti[index + 1],
                   v2Number = (size_t)ti[index + 2],
                   pNumber = (size_t)ti[index + 3];

            for (model = 0; model < pr->numberOfPartitions; ++model) {
                size_t k, states = pr->partitionData[model]->states,
                          width = pr->partitionData[model]->parsimonyLength;

                unsigned int i;

                switch (states) {
                default: {
                    /**
                     * u:    downpass state of current node
                     * p:    uppass state of parent node
                     * v_1:  downpass state of children 1 of u
                     * v_2:  downpass state of children 2 of u
                     */
                    assert(states <= 32);
                    parsimonyNumber *u[32], *v_1[32], *v_2[32], *p[32],
                        *uUppass[32];

                    for (k = 0; k < states; ++k) {
                        u[k] = &(pr->partitionData[model]
                                     ->parsVect[(width * states * uNumber) +
                                                width * k]);
                        v_1[k] = &(pr->partitionData[model]
                                       ->parsVect[(width * states * v1Number) +
                                                  width * k]);
                        v_2[k] = &(pr->partitionData[model]
                                       ->parsVect[(width * states * v2Number) +
                                                  width * k]);
                        uUppass[k] =
                            &(pr->partitionData[model]
                                  ->parsVectUppass[(width * states * uNumber) +
                                                   width * k]);
                        p[k] =
                            &(pr->partitionData[model]
                                  ->parsVectUppass[(width * states * pNumber) +
                                                   width * k]);
                    }

                    for (i = 0; i < width; ++i) {
                        parsimonyNumber x = 0, y = 0;
                        for (k = 0; k < states; ++k) {
                            x |= (~(u[k][i] & p[k][i]) & p[k][i]);
                            y |= (v_1[k][i] & v_2[k][i]);
                        }
                        y = ~y;
                        x = ~x;
                        for (k = 0; k < states; ++k) {
                            uUppass[k][i] = u[k][i];
                            uUppass[k][i] ^= ((uUppass[k][i] ^ p[k][i]) & x);
                            uUppass[k][i] ^=
                                ((uUppass[k][i] ^ (uUppass[k][i] | p[k][i])) &
                                 ((~x) & y));
                            uUppass[k][i] ^=
                                ((uUppass[k][i] ^
                                  (uUppass[k][i] |
                                   (p[k][i] & (v_1[k][i] | v_2[k][i])))) &
                                 (~(y | x)));
                        }
                    }
                }
                }
            }
        }
    }
}
unsigned int _evaluateParsimonyIterativeFastUppass(pllInstance *tr,
                                                   partitionList *pr,
                                                   int perSiteScores) {

    if (pllCostMatrix) {
        assert(0);
    }
    size_t pNumber = (size_t)tr->ti[1], qNumber = (size_t)tr->ti[2];

    size_t temNumber = 2 * (size_t)tr->mxtips - 1;
    int model;

    unsigned int bestScore = tr->bestParsimony, sum;

    if (tr->ti[0] > 4)
        _newviewParsimonyIterativeFastUppass(tr, pr, perSiteScores);

    sum = tr->parsimonyScore[pNumber] + tr->parsimonyScore[qNumber];

    for (model = 0; model < pr->numberOfPartitions; ++model) {
        size_t k, states = pr->partitionData[model]->states,
                  width = pr->partitionData[model]->parsimonyLength, i;

        switch (states) {
        case 2: {
            parsimonyNumber t_A, t_C, t_N, *left[2], *right[2];

            parsimonyNumber o_A, o_C, *tem[2];

            for (k = 0; k < 2; ++k) {
                left[k] = &(pr->partitionData[model]
                                ->parsVect[(width * 2 * qNumber) + width * k]);
                right[k] = &(pr->partitionData[model]
                                 ->parsVect[(width * 2 * pNumber) + width * k]);
                tem[k] = &(
                    pr->partitionData[model]
                        ->parsVectUppass[(width * 2 * temNumber) + width * k]);
            }

            for (i = 0; i < width; ++i) {
                t_A = left[0][i] & right[0][i];
                t_C = left[1][i] & right[1][i];

                o_A = left[0][i] | right[0][i];
                o_C = left[1][i] | right[1][i];

                t_N = ~(t_A | t_C);

                tem[0][i] = t_A | (t_N & o_A);
                tem[1][i] = t_C | (t_N & o_C);

                sum += ((unsigned int)__builtin_popcount(t_N));

                //                 if(sum >= bestScore)
                //                   return sum;
            }
        } break;
        case 4: {
            parsimonyNumber t_A, t_C, t_G, t_T, t_N, *left[4], *right[4];
            parsimonyNumber o_A, o_C, o_G, o_T, *tem[4];

            for (k = 0; k < 4; ++k) {
                left[k] = &(pr->partitionData[model]
                                ->parsVect[(width * 4 * qNumber) + width * k]);
                right[k] = &(pr->partitionData[model]
                                 ->parsVect[(width * 4 * pNumber) + width * k]);
                tem[k] = &(
                    pr->partitionData[model]
                        ->parsVectUppass[(width * 4 * temNumber) + width * k]);
            }

            for (i = 0; i < width; ++i) {
                t_A = left[0][i] & right[0][i];
                t_C = left[1][i] & right[1][i];
                t_G = left[2][i] & right[2][i];
                t_T = left[3][i] & right[3][i];

                o_A = left[0][i] | right[0][i];
                o_C = left[1][i] | right[1][i];
                o_G = left[2][i] | right[2][i];
                o_T = left[3][i] | right[3][i];

                t_N = ~(t_A | t_C | t_G | t_T);

                tem[0][i] = t_A | (t_N & o_A);
                tem[1][i] = t_C | (t_N & o_C);
                tem[2][i] = t_G | (t_N & o_G);
                tem[3][i] = t_T | (t_N & o_T);

                sum += ((unsigned int)__builtin_popcount(t_N));

                //                 if(sum >= bestScore)
                //                   return sum;
            }
        } break;
        case 20: {
            parsimonyNumber t_N, *left[20], *right[20];

            parsimonyNumber o_A[20], t_A[20], *tem[20];
            for (k = 0; k < 20; ++k) {
                left[k] = &(pr->partitionData[model]
                                ->parsVect[(width * 20 * qNumber) + width * k]);
                right[k] =
                    &(pr->partitionData[model]
                          ->parsVect[(width * 20 * pNumber) + width * k]);
                tem[k] = &(
                    pr->partitionData[model]
                        ->parsVectUppass[(width * 20 * temNumber) + width * k]);
            }

            for (i = 0; i < width; ++i) {
                t_N = 0;

                for (k = 0; k < 20; ++k) {
                    t_A[k] = left[k][i] & right[k][i];
                    o_A[k] = left[k][i] | right[k][i];
                    t_N = t_N | t_A[k];
                }

                t_N = ~t_N;

                for (k = 0; k < 20; ++k)
                    tem[k][i] = t_A[k] | (t_N & o_A[k]);
                sum += ((unsigned int)__builtin_popcount(t_N));

                //                  if(sum >= bestScore)
                //                    return sum;
            }
        } break;
        default: {
            parsimonyNumber t_N, *left[32], *right[32];

            parsimonyNumber o_A[32], t_A[32], *tem[32];
            assert(states <= 32);

            for (k = 0; k < states; ++k) {
                left[k] =
                    &(pr->partitionData[model]
                          ->parsVect[(width * states * qNumber) + width * k]);
                right[k] =
                    &(pr->partitionData[model]
                          ->parsVect[(width * states * pNumber) + width * k]);
                tem[k] = &(pr->partitionData[model]
                               ->parsVectUppass[(width * states * temNumber) +
                                                width * k]);
            }

            for (i = 0; i < width; ++i) {
                t_N = 0;

                for (k = 0; k < states; ++k) {
                    t_A[k] = left[k][i] & right[k][i];
                    o_A[k] = left[k][i] | right[k][i];
                    t_N = t_N | t_A[k];
                }

                t_N = ~t_N;

                for (k = 0; k < states; ++k)
                    tem[k][i] = t_A[k] | (t_N & o_A[k]);

                sum += ((unsigned int)__builtin_popcount(t_N));

                //                 if(sum >= bestScore)
                //                   return sum;
            }
        }
        }
    }
    uppassStatesIterativeCalculate(tr, pr);
    return sum;
}
#endif
unsigned int _evaluateParsimonyUppass(pllInstance *tr, partitionList *pr,
                                      nodeptr p, int perSiteScores) {
    // cout << "Begin _evaluateParsimonyUppass\n";
    volatile unsigned int result;
    nodeptr q = p->back;
    int *ti = tr->ti, counter = 4;

    ti[1] = p->number;
    ti[2] = q->number;
    // cout << "pNumber = " << p->number << '\n';
    // cout << "qNumber = " << q->number << '\n';
    computeTraversalInfoParsimonyUppassFull(p, ti, &counter, tr->mxtips,
                                            perSiteScores, true);
    computeTraversalInfoParsimonyUppassFull(q, ti, &counter, tr->mxtips,
                                            perSiteScores, true);

    ti[0] = counter;
    result = _evaluateParsimonyIterativeFastUppass(tr, pr, perSiteScores);
    // cout << "End _evaluateParsimonyUppass\n";
    return result;
}
void testInsertSPR(pllInstance *tr, partitionList *pr, nodeptr p, nodeptr u,
                   int perSiteScores) {
    // cout << "insert branch: " << u->number << " - " << u->back->number <<
    // '\n';

    // Uppass Score
    unsigned int mp = evaluateInsertParsimonyUppassSPR(tr, pr, p->back, u);

    if (perSiteScores) {
        // If UFBoot is enabled ...
        pllSaveCurrentTreeSprParsimony(tr, pr, mp); // run UFBoot
    }
    if (mp < tr->bestParsimony) {
        bestTreeScoreHits = 1;
    } else if (mp == tr->bestParsimony) {
        bestTreeScoreHits++;
    }
    if ((mp < tr->bestParsimony) ||
        ((mp == tr->bestParsimony) &&
         (random_double() <= 1.0 / bestTreeScoreHits))) {
        tr->bestParsimony = mp;
        tr->TBR_insertBranch1 = u;
        tr->TBR_removeBranch = p;
    }
}
void testInsertTBR(pllInstance *tr, partitionList *pr, nodeptr p, nodeptr q,
                   nodeptr r, int perSiteScores) {
    // cout << "insert branch: " << p->number << " - " << p->back->number
    //     << '\n';
    // cout << "insert branch: " << q->number << " - " << q->back->number
    //     << '\n';
    cnt++;
    // Uppass Score
    unsigned int mp = evaluateInsertParsimonyUppassTBR(tr, pr, p, q);
    // cout << "Insert MP = " << mp << '\n';

    if (perSiteScores) {
        // If UFBoot is enabled ...
        pllSaveCurrentTreeSprParsimony(tr, pr, mp); // run UFBoot
    }
    if (mp < tr->bestParsimony) {
        bestTreeScoreHits = 1;
    } else if (mp == tr->bestParsimony) {
        bestTreeScoreHits++;
    }
    if ((mp < tr->bestParsimony) ||
        ((mp == tr->bestParsimony) &&
         (random_double() <= 1.0 / bestTreeScoreHits))) {
        tr->bestParsimony = mp;
        tr->TBR_insertBranch1 = p;
        tr->TBR_insertBranch2 = q;
        tr->TBR_removeBranch = r;
    }
}
// void testInsertTBRCorrectness(pllInstance *tr, partitionList *pr, nodeptr p,
//                               nodeptr q, nodeptr r, int perSiteScores) {
//     cout << "insert branch: " << p->number << " - " << p->back->number <<
//     '\n'; cout << "insert branch: " << q->number << " - " << q->back->number
//     << '\n';
//
//     // Uppass Score
//     unsigned int uppassMP = evaluateInsertParsimonyUppassTBR(tr, pr, p, q);
//
//     // Downpass Score
//     nodeptr p1 = p->back;
//     nodeptr q1 = q->back;
//     nodeptr r1 = r->back;
//
//     // Try to connect
//     r->next->back = p;
//     r->next->next->back = p1;
//     p->back = r->next;
//     p1->back = r->next->next;
//
//     r1->next->back = q;
//     r1->next->next->back = q1;
//     q->back = r1->next;
//     q1->back = r1->next->next;
//
//     unsigned int mp = _evaluateParsimony(tr, pr, r, PLL_TRUE, PLL_FALSE);
//     // Rollback
//     r->next->back = r->next->next->back = NULL;
//     r1->next->back = r1->next->next->back = NULL;
//     p->back = p1;
//     p1->back = p;
//     q->back = q1;
//     q1->back = q;
//
//     // cout << "Uppass Score: " << uppassMP << '\n';
//     // cout << "Real Score: " << mp << '\n';
//     assert(uppassMP == mp);
// }
// void testInsertSPRCorrectness(pllInstance *tr, partitionList *pr, nodeptr p,
//                               nodeptr u) {
//     cout << "insert branch: " << u->number << " - " << u->back->number <<
//     '\n';
//
//     // Uppass Score
//     unsigned int uppassMP =
//         evaluateInsertParsimonyUppassSPR(tr, pr, p->back, u);
//
//     // Downpass Score
//     nodeptr v = u->back;
//
//     // Try to insert
//     p->next->back = u;
//     p->next->next->back = v;
//     u->back = p->next;
//     v->back = p->next->next;
//
//     unsigned int mp = _evaluateParsimony(tr, pr, p, PLL_TRUE, PLL_FALSE);
//     // Rollback
//     p->next->back = p->next->next->back = NULL;
//     u->back = v;
//     v->back = u;
//
//     cout << "Uppass Score: " << uppassMP << '\n';
//     cout << "Real Score: " << mp << '\n';
//     assert(uppassMP == mp);
// }

void traverseInsertBranchesTBRQ(pllInstance *tr, partitionList *pr, nodeptr p,
                                nodeptr q, nodeptr r, int mintrav, int maxtrav,
                                int perSiteScores) {

    if (mintrav <= 0) {
        testInsertTBR(tr, pr, p, q, r, perSiteScores);
    }
    /* traverse the q subtree */
    if (!isTip(q->number, tr->mxtips) && maxtrav >= 1) {
        traverseInsertBranchesTBRQ(tr, pr, p, q->next->back, r, mintrav - 1,
                                   maxtrav - 1, perSiteScores);
        traverseInsertBranchesTBRQ(tr, pr, p, q->next->next->back, r,
                                   mintrav - 1, maxtrav - 1, perSiteScores);
    }
}
void traverseInsertBranchesTBRP(pllInstance *tr, partitionList *pr, nodeptr p,
                                nodeptr q, nodeptr r, int mintrav, int maxtrav,
                                int perSiteScores) {
    if (maxtrav < 0) {
        return;
    }
    traverseInsertBranchesTBRQ(tr, pr, p, q, r, mintrav, maxtrav,
                               perSiteScores);
    /* traverse the p subtree */
    if (!isTip(p->number, tr->mxtips) && maxtrav >= 1) {
        traverseInsertBranchesTBRP(tr, pr, p->next->back, q, r, mintrav - 1,
                                   maxtrav - 1, perSiteScores);
        traverseInsertBranchesTBRP(tr, pr, p->next->next->back, q, r,
                                   mintrav - 1, maxtrav - 1, perSiteScores);
    }
}
void traverseInsertBranchesTBRQ(pllInstance *tr, partitionList *pr, nodeptr p,
                                nodeptr q, nodeptr r, int perSiteScores) {
    testInsertTBR(tr, pr, p, q, r, perSiteScores);
    /* traverse the q subtree */
    if (!isTip(q->number, tr->mxtips)) {
        traverseInsertBranchesTBRQ(tr, pr, p, q->next->back, r, perSiteScores);
        traverseInsertBranchesTBRQ(tr, pr, p, q->next->next->back, r,
                                   perSiteScores);
    }
}
void traverseInsertBranchesTBRP(pllInstance *tr, partitionList *pr, nodeptr p,
                                nodeptr q, nodeptr r, int perSiteScores) {
    traverseInsertBranchesTBRQ(tr, pr, p, q, r, perSiteScores);
    /* traverse the p subtree */
    if (!isTip(p->number, tr->mxtips)) {
        traverseInsertBranchesTBRP(tr, pr, p->next->back, q, r, perSiteScores);
        traverseInsertBranchesTBRP(tr, pr, p->next->next->back, q, r,
                                   perSiteScores);
    }
}
void traverseInsertBranchesSPR(pllInstance *tr, partitionList *pr, nodeptr p,
                               nodeptr u, int mintrav, int maxtrav,
                               int perSiteScores) {
    if (maxtrav < 0) {
        return;
    }
    if (mintrav <= 0) {
        testInsertSPR(tr, pr, p, u, perSiteScores);
    }
    if (u->number > tr->mxtips && maxtrav >= 1) {
        traverseInsertBranchesSPR(tr, pr, p, u->next->back, mintrav - 1,
                                  maxtrav - 1, perSiteScores);
        traverseInsertBranchesSPR(tr, pr, p, u->next->next->back, mintrav - 1,
                                  maxtrav - 1, perSiteScores);
    }
}
void traverseInsertBranchesSPR(pllInstance *tr, partitionList *pr, nodeptr p,
                               nodeptr u, int perSiteScores) {
    testInsertSPR(tr, pr, p, u, perSiteScores);
    if (u->number > tr->mxtips) {
        traverseInsertBranchesSPR(tr, pr, p, u->next->back, perSiteScores);
        traverseInsertBranchesSPR(tr, pr, p, u->next->next->back,
                                  perSiteScores);
    }
}
void copySingleGlobalToLocalUppass(partitionList *pr, int uNumber) {
    if (isUppassCopied[uNumber]) {
        return;
    }
    isUppassCopied[uNumber] = true;
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t states = pr->partitionData[model]->states,
               width = pr->partitionData[model]->parsimonyLength;
        for (int i = width * states * uNumber;
             i < width * states * (uNumber + 1); ++i) {
            pr->partitionData[model]->parsVectUppassLocal[i] =
                pr->partitionData[model]->parsVectUppass[i];
        }
    }
}
void copyGlobalToLocalUppass(partitionList *pr, int limNodes) {
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        size_t states = pr->partitionData[model]->states,
               width = pr->partitionData[model]->parsimonyLength;
        for (int i = 0; i < width * states * limNodes; ++i) {
            pr->partitionData[model]->parsVectUppassLocal[i] =
                pr->partitionData[model]->parsVectUppass[i];
        }
    }
}

#if (defined(__SSE3) || defined(__AVX))
/**
 * WARNING: Haven't tested yet.
 */
inline bool equalStatesCmp(parsimonyNumber *uStatesVec,
                           parsimonyNumber *vStatesVec, std::size_t states) {
//    int posU = 0, posV = 0;
//    for (int k = 0; k < states; ++k) {
//        for (int i = 0; i < INTS_PER_VECTOR; ++i) {
//            if (uStatesVec[posU + i] != vStatesVec[posV + i]) {
//                return false;
//            }
//        }
//        posU += stepU;
//        posV += stepV;
//    }
    return std::equal(uStatesVec, uStatesVec + (states * INTS_PER_VECTOR), vStatesVec);
}
/**
 * Set u = v.
 */
inline void setStatesVec(parsimonyNumber *uStatesVec,
                         parsimonyNumber *vStatesVec, int stepU, int stepV,
                         int &states) {
    int posU = 0, posV = 0;
    for (int k = 0; k < states; ++k) {
        for (int i = 0; i < INTS_PER_VECTOR; ++i) {
            uStatesVec[posU + i] = vStatesVec[posV + i];
        }
        posU += stepU;
        posV += stepV;
    }
}
/**
 * WARNING: Pls check!
 */
void uppassLeafNodeCalculate(parsimonyNumber *uVec, parsimonyNumber *uUpVec,
                             parsimonyNumber *pUpVec, std::size_t states) {
    // cout << "Begin uppassLeafNodeCalculate\n";
    INT_TYPE
    allOne = SET_ALL_BITS_ONE;
    switch (states) {
    default: {
        /**
         * u:    downpass state of current leaf
         * p:    uppass state of parent node
         */
        assert(states <= 32);

        INT_TYPE x = SET_ALL_BITS_ZERO, u_k[32], p_k[32];
        // parsimonyNumber x = 0;
        std::size_t posU, posUUp, posPUp;
        posU = posPUp = 0;
        for (int k = 0; k < states; ++k) {
            u_k[k] = VECTOR_LOAD((CAST)(&uVec[posU]));
            p_k[k] = VECTOR_LOAD((CAST)(&pUpVec[posPUp]));
            x = VECTOR_BIT_OR(
                x, VECTOR_BIT_AND(
                       VECTOR_AND_NOT(VECTOR_BIT_AND(u_k[k], p_k[k]), allOne),
                       p_k[k]));
            // x |= (~(uVec[posU] & pUpVec[posPUp]) & pUpVec[posPUp]);
//            posU += stepU;
//            posPUp += stepPUp;
            posU = next_state(posU);
            posPUp = next_state(posPUp);
        }
        x = VECTOR_AND_NOT(x, allOne);
        // x = ~x;
        // posUUp = posPUp = 0;
        posU = posUUp = posPUp = 0;
        for (int k = 0; k < states; ++k) {
            u_k[k] = VECTOR_BIT_XOR(
                u_k[k], VECTOR_BIT_AND(VECTOR_BIT_XOR(u_k[k], p_k[k]), x));
            VECTOR_STORE((CAST)(&uUpVec[posUUp]), u_k[k]);
            // uUpVec[posUUp] = uVec[posU];
            // uUpVec[posUUp] ^= ((uUpVec[posUUp] ^ pUpVec[posPUp]) & x);
//            posU += stepU;
//            posUUp += stepUUp;
//            posPUp += stepPUp;
            posU = next_state(posU);
            posUUp = next_state(posUUp);
            posPUp = next_state(posPUp);
        }
    }
    }
    // cout << "End uppassLeafNodeCalculate\n";
}
/**
 * Do Fitch downpass algorithm calculate uVec downpass from 2 of its children v1
 * and v2.
 *
 * Return parsimonyNumber t_N indicate whether any column is AND of its 2
 * children or OR of its 2 children.
 *
 * If bit i-th is bit 1 -> OR of its 2 children (Cost 1) at column i.
 * If bit i-th is bit 0 -> AND of its 2 children (Cost 0) at column i.
 *
 * -> Number of bit 1 of t_N is the total cost of all the columns represented in
 * t_N.
 */
INT_TYPE downpassCalculate(parsimonyNumber *uVec, parsimonyNumber *v1Vec,
                           parsimonyNumber *v2Vec, std::size_t states) {
    // cout << "Begin downpassCalculate\n";
    INT_TYPE
    allOne = SET_ALL_BITS_ONE;
    INT_TYPE t_N = SET_ALL_BITS_ZERO;
    // parsimonyNumber t_N = 0;
    switch (states) {
    /* TODO: Add case 2, 4, 20 */
    default: {
        // assert(states <= 32)
        INT_TYPE s_l, s_r, t_A[32], o_A[32];
        // parsimonyNumber o_A[32], t_A[32];
        std::size_t posV1 = 0, posV2 = 0;
        for (int k = 0; k < states; ++k) {
            s_l = VECTOR_LOAD((CAST)(&v1Vec[posV1]));
            s_r = VECTOR_LOAD((CAST)(&v2Vec[posV2]));
            o_A[k] = VECTOR_BIT_OR(s_l, s_r);
            t_A[k] = VECTOR_BIT_AND(s_l, s_r);
            // o_A[k] = v1Vec[posV1] | v2Vec[posV2];
            // t_A[k] = v1Vec[posV1] & v2Vec[posV2];

            t_N = VECTOR_BIT_OR(t_N, t_A[k]);
            // t_N |= t_A[k];
//            posV1 += stepV1;
//            posV2 += stepV2;
            posV1 = next_state(posV1);
            posV2 = next_state(posV2);
        }
        t_N = VECTOR_AND_NOT(t_N, allOne);
        // t_N = ~t_N;
        std::size_t posU = 0;
        for (int k = 0; k < states; ++k) {
            VECTOR_STORE((CAST)(&uVec[posU]),
                         VECTOR_BIT_OR(t_A[k], VECTOR_BIT_AND(t_N, o_A[k])));
            // uVec[posU] = t_A[k] | (t_N & o_A[k]);
//            posU += stepU;
            posU = next_state(posU);
        }
    }
    }
    // cout << "End downpassCalculate\n";
    return t_N;
}
/**
 * WARNING: Pls check.
 */
void uppassInnerNodeCalculate(parsimonyNumber *uVec, parsimonyNumber *v1Vec,
                              parsimonyNumber *v2Vec, parsimonyNumber *uUpVec,
                              parsimonyNumber *pUpVec, std::size_t states) {
    // cout << "Begin uppassInnerNodeCalculate\n";
    INT_TYPE
    allOne = SET_ALL_BITS_ONE;
    switch (states) {
    default: {
        /**
         * u:    downpass state of current node
         * p:    uppass state of parent node
         * v_1:  downpass state of children 1 of u
         * v_2:  downpass state of children 2 of u
         */
        // assert(states <= 32);

        INT_TYPE x = SET_ALL_BITS_ZERO, y = SET_ALL_BITS_ZERO, u_k[32], p_k[32],
                 v_1k[32], v_2k[32], u_up;
        // parsimonyNumber x = 0, y = 0;
        std::size_t posU, posV1, posV2, posUUp, posPUp;
        posU = posV1 = posV2 = posPUp = 0;
        for (int k = 0; k < states; ++k) {
            u_k[k] = VECTOR_LOAD((CAST)(&uVec[posU]));
            p_k[k] = VECTOR_LOAD((CAST)(&pUpVec[posPUp]));
            v_1k[k] = VECTOR_LOAD((CAST)(&v1Vec[posV1]));
            v_2k[k] = VECTOR_LOAD((CAST)(&v2Vec[posV2]));
//            x = VECTOR_BIT_OR(
//                x, VECTOR_AND_NOT(VECTOR_BIT_AND(u_k[k], p_k[k]), p_k[k]));
            x = VECTOR_BIT_OR(
                    x, VECTOR_AND_NOT(u_k[k], p_k[k]));
            y = VECTOR_BIT_OR(y, VECTOR_BIT_AND(v_1k[k], v_2k[k]));
            // x |= (~(uVec[posU] & pUpVec[posPUp]) & pUpVec[posPUp]);
            // y |= (v1Vec[posV1] & v2Vec[posV2]);
//            posU += stepU;
//            posV1 += stepV1;
//            posV2 += stepV2;
//            posPUp += stepPUp;
            posU = next_state(posU);
            posV1 = next_state(posV1);
            posV2 = next_state(posV2);
            posPUp = next_state(posPUp);
        }
        posU = posV1 = posV2 = posUUp = posPUp = 0;
        // posUUp = 0;
        for (int k = 0; k < states; ++k) {
            u_up = u_k[k];

            u_up = VECTOR_BIT_XOR(u_up,
                                  VECTOR_BIT_AND(VECTOR_BIT_XOR(u_up, p_k[k]),
                                                 VECTOR_AND_NOT(x, allOne)));
            u_up = VECTOR_BIT_XOR(
                u_up, VECTOR_BIT_AND(
                          VECTOR_BIT_XOR(u_up, VECTOR_BIT_OR(u_up, p_k[k])),
                          VECTOR_BIT_AND(x, VECTOR_AND_NOT(y, allOne))));
            u_up = VECTOR_BIT_XOR(
                u_up,
                VECTOR_BIT_AND(
                    VECTOR_BIT_XOR(
                        u_up, VECTOR_BIT_OR(
                                  u_up, VECTOR_BIT_AND(
                                            p_k[k],
                                            VECTOR_BIT_OR(v_1k[k], v_2k[k])))),
                    VECTOR_BIT_AND(x, y)));
            VECTOR_STORE((CAST)(&uUpVec[posUUp]), u_up);
            // uUpVec[posUUp] = uVec[posU];
            // uUpVec[posUUp] ^= ((uUpVec[posUUp] ^ pUpVec[posPUp]) & (~x));
            // uUpVec[posUUp] ^=
            //     ((uUpVec[posUUp] ^ (uUpVec[posUUp] | pUpVec[posPUp])) &
            //      (x & (~y)));
            // uUpVec[posUUp] ^=
            //     ((uUpVec[posUUp] ^
            //       (uUpVec[posUUp] |
            //        (pUpVec[posPUp] & (v1Vec[posV1] | v2Vec[posV2])))) &
            //      (x & y));
//            posU += stepU;
//            posV1 += stepV1;
//            posV2 += stepV2;
//            posUUp += stepUUp;
//            posPUp += stepPUp;
            posU = next_state(posU);
            posV1 = next_state(posV1);
            posV2 = next_state(posV2);
            posUUp = next_state(posUUp);
            posPUp = next_state(posPUp);
        }
    }
    }
    // cout << "End uppassInnerNodeCalculate\n";
}
/**
 * STEP 2b
 * Update new uppass calculated to parsVectUppassLocal
 */
void dfsRecalculateUppassLocal(nodeptr u, parsimonyNumber *pUpVec, pInfo *prModel, partitionList *pr, std::size_t w,
                               std::size_t mxTips) {
    // cout << "Begin dfsRecalculateUppassLocal\n";
    if (!inRadiusRange[u->number]) {
        return;
    }
    copySingleGlobalToLocalUppass(pr, u->number);
    int width = prModel->parsimonyLength;
    int states = prModel->states;
//    int posU = width * states * u->number + w;
    std::size_t posU = table_index(width, states, u->number, 0, w);
    if (u->number <= mxTips) {
        uppassLeafNodeCalculate(&(prModel->parsVect[posU]),
                                &(prModel->parsVectUppassLocal[posU]), pUpVec, states);
    } else {
        nodeptr v1 = u->next->back;
        nodeptr v2 = u->next->next->back;
        uppassInnerNodeCalculate(
            &(prModel->parsVect[posU]),
//            &(prModel->parsVect[width * states * v1->number + w]),
            &(prModel->parsVect[table_index(width, states, v1->number, 0, w)]),
//            &(prModel->parsVect[width * states * v2->number + w]),
            &(prModel->parsVect[table_index(width, states, v2->number, 0, w)]),
            &(prModel->parsVectUppassLocal[posU]), pUpVec, states);
        if ((inRadiusRange[v1->number] || inRadiusRange[v2->number]) &&
            !equalStatesCmp(&(prModel->parsVectUppassLocal[posU]),
                            &(prModel->parsVectUppass[posU]), states)) {
            dfsRecalculateUppassLocal(v1, &(prModel->parsVectUppassLocal[posU]), prModel, pr, w, mxTips);
            dfsRecalculateUppassLocal(v2, &(prModel->parsVectUppassLocal[posU]), prModel, pr, w, mxTips);
        }
    }
    // cout << "End dfsRecalculateUppassLocal\n";
}
/**
 * STEP 2b
 * Update new uppass calculated to parsVectUppass
 */
void dfsRecalculateUppass(nodeptr u, parsimonyNumber *pUpVec, pInfo *prModel,
                          partitionList *pr, std::size_t w, std::size_t mxTips) {
    int width = prModel->parsimonyLength;
    int states = prModel->states;
//    int posU = width * states * u->number + w;
    std::size_t posU = table_index(width, states, u->number, 0, w);
    if (u->number <= mxTips) {
        uppassLeafNodeCalculate(&(prModel->parsVect[posU]),
                                &(prModel->parsVectUppass[posU]), pUpVec, states);
    } else {
        copySingleGlobalToLocalUppass(pr, u->number);
        nodeptr v1 = u->next->back;
        nodeptr v2 = u->next->next->back;
        uppassInnerNodeCalculate(
            &(prModel->parsVect[posU]),
//            &(prModel->parsVect[width * states * v1->number + w]),
            &(prModel->parsVect[table_index(width, states, v1->number, 0, w)]),
//            &(prModel->parsVect[width * states * v2->number + w]),
            &(prModel->parsVect[table_index(width, states, v2->number, 0, w)]),
            &(prModel->parsVectUppass[posU]), pUpVec, states);
        if (!equalStatesCmp(&(prModel->parsVectUppassLocal[posU]),
                            &(prModel->parsVectUppass[posU]), states)) {
            dfsRecalculateUppass(v1, &(prModel->parsVectUppass[posU]), prModel, pr, w, mxTips);
            dfsRecalculateUppass(v2, &(prModel->parsVectUppass[posU]), prModel, pr, w, mxTips);
        }
    }
}
/**
 * Use incremental downpass and uppass, recalculate only nodes need to be
 * recalculated. Nm and its 2 children must be still connected. Can be both used
 * for TBR and SPR?.
 *
 * @param Nm Branch Nm - Nx (Nx is Nm->back) is the remove branch. It COULD be
 * leaf branch or inner branch (as this applys for both TBR and SPR).
 * @param Nz, Ns 2 vertices that previously connected to Nx.
 *
 * @return sum of MP score of 2 divided trees.
 */
unsigned int recalculateDownpassAndUppass(pllInstance *tr, partitionList *pr,
                                          nodeptr Nm, nodeptr Nz, nodeptr Ns) {
    // cout << "Begin recalculateDownpassAndUppass\n";
    // cout << "Nz = " << Nz->number << '\n';
    if (depth[Ns->number] < depth[Nz->number]) {
        swap(Ns, Nz);
    }
    nodeptr Nx = Nm->back;
    /* TODO: Mark which nodes are changed and reassign "global" to "local" on
     * only those nodes
     */
    // copyGlobalToLocalUppass(pr, 2 * tr->mxtips - 1);
    /* WARNING: What if Nm is a leaf? Will the score be 0? */
    unsigned int scoreClippedSubtree = tr->parsimonyScore[Nm->number];
    // cout << "CUR scoreClippedSubtree = " << scoreClippedSubtree << '\n';
    unsigned int scoreMainSubtree = 0;

    if (depth[Nx->number] == 0 && depth[Nm->number] == 0) {
        scoreMainSubtree = tr->parsimonyScore[Nx->number];
    } else if (depth[Nx->number] == 0 && depth[Nm->number] == 1) {
        /* WARNING: What if Nz or Ns is a leaf? Will the score be 0? */
        scoreMainSubtree =
            tr->parsimonyScore[Nz->number] + tr->parsimonyScore[Ns->number];
        /* score has to be updated below */
    } else {
        scoreMainSubtree = oldScore - tr->parsimonyScore[Nx->number] +
                           tr->parsimonyScore[Ns->number];
        // cout << "CUR scoreMainSubtree = " << scoreMainSubtree << '\n';
        /* score has to be updated below */
    }
    for (int model = 0; model < pr->numberOfPartitions; ++model) {
        std::size_t states = pr->partitionData[model]->states,
            width = pr->partitionData[model]->parsimonyLength;
        auto idx = [width, states](std::size_t state_id, std::size_t col_id) -> std::size_t {
            return table_index(width, states, 0, state_id, col_id);
        };
        auto init = [width, states](std::size_t node_id, std::size_t col_id = 0) -> std::size_t {
            return table_index(width, states, node_id, 0, col_id);
        };

        for (int w = 0; w < width; w += INTS_PER_VECTOR) {
//            int posNm = width * states * Nm->number + w;
//            int posNs = width * states * Ns->number + w;
//            int posNx = width * states * Nx->number + w;
//            int posNz = width * states * Nz->number + w;
            std::size_t posNm = init(Nm->number, w);
            std::size_t posNs = init(Ns->number, w);
            std::size_t posNx = init(Nx->number, w);
            std::size_t posNz = init(Nz->number, w);
            /* STEP 2d: Recalculate Nm and its subtree
             * if old (global) uppass != old (global) downpass
             * (Because its new uppass is now its downpass) */
            if (!equalStatesCmp(
                    &(pr->partitionData[model]->parsVect[posNm]),
                    &(pr->partitionData[model]->parsVectUppass[posNm]), states)) {
                // /**
                //  * Set new (local) uppass of Nm = old (global) downpass of Nm.
                //  * (As Nm is now the root of its subtree).
                //  * This MUST be done as SPR would use Nm's new (local) uppass.
                //  */
                // setStatesVec(
                //     &(pr->partitionData[model]->parsVectUppassLocal[posNm]),
                //     &(pr->partitionData[model]->parsVect[posNm]), width, width,
                //     states);
                /* Nm might be LEAF or might have 2 children. */
                if (Nm->number > tr->mxtips) {
                    dfsRecalculateUppassLocal(
                        Nm->next->back,
                        &(pr->partitionData[model]->parsVect[posNm]), pr->partitionData[model], pr, w, tr->mxtips);
                    dfsRecalculateUppassLocal(
                        Nm->next->next->back,
                        &(pr->partitionData[model]->parsVect[posNm]), pr->partitionData[model], pr, w, tr->mxtips);
                }
            }
            /* Nx MUST be parent of Nm or Nx-Nm is the root branch */
            assert(depth[Nm->number] >= depth[Nx->number]);
            if (depth[Nx->number] == 0 && depth[Nm->number] == 0) {
                // cout << "Remove branch is root branch\n";
                /**
                 * Nx - Nm is the root branch
                 * Skip STEP 1, 2a, 2c
                 */
                /**
                 * If old downpass (= new uppass) != old uppass of Nx
                 * -> Recalculate its subtree
                 */
                if (!equalStatesCmp(
                        &(pr->partitionData[model]->parsVect[posNx]),
                        &(pr->partitionData[model]->parsVectUppass[posNx]), states)) {
                    // /**
                    //  * Set new (local) uppass of Nx = old (global) downpass
                    //  * of Nx. (As Nx is now the root of its subtree).
                    //  */
                    // setStatesVec(
                    //     &(pr->partitionData[model]->parsVectUppassLocal[posNx]),
                    //     &(pr->partitionData[model]->parsVect[posNx]), width,
                    //     width, states);
                    /* Nx must have 2 children (Ns and Nz). */
                    dfsRecalculateUppassLocal(
                        Ns,
                        &(pr->partitionData[model]->parsVect[posNx]), pr->partitionData[model], pr, w, tr->mxtips);
                    dfsRecalculateUppassLocal(
                        Nz,
                        &(pr->partitionData[model]->parsVect[posNx]), pr->partitionData[model], pr, w, tr->mxtips);
                }
                /* Continue to the next set of 32 columns */
                continue;
            } else if (depth[Nx->number] == 0 && depth[Nm->number] == 1) {
                /* Nx-Nz must be the root branch here */
                assert(depth[Nz->number] == 0);

                /* Calculate new downpass (= new uppass) of the new root branch
                 * Ns-Nz */
                parsimonyNumber *rootDownpass = NULL;
                rax_posix_memalign((void **)&(rootDownpass), PLL_BYTE_ALIGNMENT,
                                   (size_t)INTS_PER_VECTOR * states *
                                       sizeof(parsimonyNumber));
                // parsimonyNumber *rootDownpass = new
                // alignas(PLL_BYTE_ALIGNMENT)
                //     parsimonyNumber[states * INTS_PER_VECTOR];
                INT_TYPE isUnionDownpass = downpassCalculate(
                    rootDownpass, &(pr->partitionData[model]->parsVect[posNs]),
                    &(pr->partitionData[model]->parsVect[posNz]), states);

                /* Update scoreMainSubtree */
                // scoreMainSubtree += __builtin_popcount(isUnionDownpass);
                {
                    unsigned int counts[INTS_PER_VECTOR]
                        __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                    VECTOR_STORE((CAST)counts, isUnionDownpass);

                    for (int ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                        scoreMainSubtree += __builtin_popcount(counts[ptr]);
                    }
                }

                /* STEP 2c: If old uppass of Nx != new downpass (new uppass) of
                 * root branch Ns-Nz, do dfsRecalculateUppass on Ns */
                if (!equalStatesCmp(
                        &(pr->partitionData[model]->parsVectUppass[posNx]),
                        rootDownpass, states)) {
                    dfsRecalculateUppassLocal(Ns, rootDownpass,
                                              pr->partitionData[model], pr, w,
                                              tr->mxtips);
                }

                /* dfsRecalculateUppass on Nz
                 * (As we don't save the value of old uppass of old root branch
                 * -> Can't compare)
                 */
                /* TODO: Add if */
                dfsRecalculateUppassLocal(Nz, rootDownpass,
                                          pr->partitionData[model], pr, w,
                                          tr->mxtips);
                /* Continue to the next set of 32 columns */
                // delete[] rootDownpass;
                if (rootDownpass != NULL) {
                    rax_free(rootDownpass);
                    rootDownpass = NULL;
                }
                continue;
            }
            /**
             * old (global) downpass of N_s != old (global) downpass of N_x
             * -> Do STEP 1, 2a, 2b
             */
            if (!equalStatesCmp(&(pr->partitionData[model]->parsVect[posNs]),
                                &(pr->partitionData[model]->parsVect[posNx]), states)) {
                nodeptr u = Nz, lastU = NULL;
                if (depth[u->next->back->number] <
                    depth[u->next->next->back->number]) {
                    u = u->next;
                } else {
                    u = u->next->next;
                }
                parsimonyNumber *uDownpass, *lastUDownpass = NULL;
                struct pair2 {
                    nodeptr v;
                    parsimonyNumber *vDownpass;
                    pair2(nodeptr _v, parsimonyNumber *_vDownpass)
                        : v(_v), vDownpass(_vDownpass) {}
                };

                /* HACK: Is this good? */
                vector<pair2> nodesRecalculated;
                nodesRecalculated.reserve(16);

                bool rootDownpassChanged = false;

                // cout << "STEP 1\n";
                /* STEP 1: Do incremental downpass, from Nz up to the root while
                 * still needs recalculated */
                while (true) {
                    assert(u->number > tr->mxtips);
                    nodeptr v1 = u->next->back;
                    nodeptr v2 = u->next->next->back;
                    parsimonyNumber *v1Downpass, *v2Downpass;
//                    int stepV1, stepV2;
                    if (lastU != NULL && v1->number == lastU->number) {
                        assert(lastUDownpass != NULL);
                        v1Downpass = lastUDownpass;
//                        v2Downpass = &(
//                            pr->partitionData[model]
//                                ->parsVect[(width * states * v2->number) + w]);
                        v2Downpass = &(
                                pr->partitionData[model]
                                        ->parsVect[init(v2->number, w)]);
//                        stepV1 = INTS_PER_VECTOR;
//                        stepV2 = width;
                    } else if (lastU != NULL && v2->number == lastU->number) {
                        assert(lastUDownpass != NULL);
//                        v1Downpass = &(
//                            pr->partitionData[model]
//                                ->parsVect[(width * states * v1->number) + w]);
                        v1Downpass = &(
                                pr->partitionData[model]
                                        ->parsVect[init(v1->number, w)]);
                        v2Downpass = lastUDownpass;
//                        stepV1 = width;
//                        stepV2 = INTS_PER_VECTOR;
                    } else {
                        /* Happens only the first time (i.e. lastU = NULL)
                         */
                        assert(lastU == NULL);
//                        v1Downpass = &(
//                            pr->partitionData[model]
//                                ->parsVect[(width * states * v1->number) + w]);
//                        v2Downpass = &(
//                            pr->partitionData[model]
//                                ->parsVect[(width * states * v2->number) + w]);
                        v1Downpass = &(
                                pr->partitionData[model]
                                        ->parsVect[init(v1->number, w)]);
                        v2Downpass = &(
                                pr->partitionData[model]
                                        ->parsVect[init(v2->number, w)]);
//                        stepV1 = stepV2 = width;
                    }

                    /* Fitch's Algo */
                    rax_posix_memalign((void **)&(uDownpass),
                                       PLL_BYTE_ALIGNMENT,
                                       (size_t)INTS_PER_VECTOR * states *
                                           sizeof(parsimonyNumber));
                    // uDownpass = new alignas(PLL_BYTE_ALIGNMENT)
                    //     parsimonyNumber[states * INTS_PER_VECTOR];
                    INT_TYPE isUnionDownpass = downpassCalculate(
                        uDownpass, v1Downpass, v2Downpass, states);
                    /* Update score difference when recalculate downpass */
                    // cout << "scoreMainSubtree BEGIN = " <<
                    // scoreMainSubtree
                    //      << '\n';
                    // cout << "Score decrease: "
                    //      << pr->partitionData[model]
                    //             ->scoreIncrease[width * u->number + w]
                    //      << '\n';

                    for (int ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                        scoreMainSubtree -=
                            pr->partitionData[model]
                                ->scoreIncrease[width * u->number + w + ptr];
                    }

                    // cout << "Score increase: "
                    //      << __builtin_popcount(isUnionDownpass) << '\n';
                    // scoreMainSubtree += __builtin_popcount(isUnionDownpass);
                    {
                        unsigned int counts[INTS_PER_VECTOR]
                            __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                        VECTOR_STORE((CAST)counts, isUnionDownpass);

                        for (int ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                            scoreMainSubtree += __builtin_popcount(counts[ptr]);
                        }
                    }

                    // cout v< "scoreMainSubtree AFTER = " <<
                    // scoreMainSubtree
                    //      << '\n';

                    /**
                     * Uppass of u MUST be recalculated in cases:
                     *  - Downpass of u changed
                     *  - Downpass of 2 children v1, v2 of u changed
                     *  - Uppass of parent p of u changed
                     */
                    nodesRecalculated.emplace_back(pair2{u, uDownpass});
                    /**
                     * NOTE: u is root branch
                     * IF equalStatesCmp().. = TRUE
                     *   -> Uppass (also Downpass) of root doesn't change.
                     * But still have to recalculate downpass of root
                     * because the downpass of root isn't saved.
                     *   -> Don't have to dfs_recalculate_uppass on u->back
                     *   -> dfs_recalculate_uppass on u
                     * IF equalStatesCmp().. = FALSE
                     *   -> Recalculate downpass (also uppass) of root
                     *   -> dfs_recalculate_uppass on both u and u->back
                     */
                    /* If downpass of u doesn't change, break the while
                     * loop */
                    if (equalStatesCmp(
                            uDownpass,
                            &(pr->partitionData[model]
                                  ->parsVect[init(u->number, w)]), states)) {

                        // cout << "u->back->number = " << u->back->number
                        // <<
                        // '\n';
                        break;
                    }
                    /* if u is root branch, break the while loop */
                    if (depth[u->number] == 0) {
                        /* As equalStatesCmp()... above = FALSE, have to
                         * dfsRecalculateUppass on u->back too */
                        rootDownpassChanged = true;
                        break;
                    }
                    lastU = u;
                    lastUDownpass = uDownpass;
                    u = uppass_par[u->number];
                }
                /* STEP 2a */
                parsimonyNumber *rootDownpass = NULL;
                if (depth[u->number] == 0) {
                    assert(depth[u->back->number] == 0);
                    /**
                     * NOTE: Calculate root downpass from u and u->back. Use
                     * root downpass to calculate uppass of u and u->back.
                     * Do dfs_recalculate_uppass at u->back too
                     */

                    /* Calculate new downpass (= new uppass) of the old root
                     */
                    rax_posix_memalign((void **)&(rootDownpass),
                                       PLL_BYTE_ALIGNMENT,
                                       (size_t)INTS_PER_VECTOR * states *
                                           sizeof(parsimonyNumber));
                    // rootDownpass = new alignas(PLL_BYTE_ALIGNMENT)
                    //     parsimonyNumber[states * INTS_PER_VECTOR];
                    INT_TYPE isUnionDownpassRoot = downpassCalculate(
                        rootDownpass, uDownpass,
                        &(pr->partitionData[model]
                              ->parsVect[init(u->back->number, w)]), states);

                    if (rootDownpassChanged == true) {
                        // cout << "rootDownpassChanged\n";
                        /* Update scoreMainSubtree */
                        /* Subtract the old root downpass cost */

                        for (int ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                            scoreMainSubtree -=
                                pr->partitionData[model]->scoreIncrease
                                    [width * (2 * tr->mxtips - 1) + w + ptr];
                        }
                        /* Add the new increased cost */
                        // scoreMainSubtree +=
                        //     __builtin_popcount(isUnionDownpassRoot);
                        {
                            unsigned int counts[INTS_PER_VECTOR]
                                __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

                            VECTOR_STORE((CAST)counts, isUnionDownpassRoot);

                            for (int ptr = 0; ptr < INTS_PER_VECTOR; ++ptr) {
                                scoreMainSubtree +=
                                    __builtin_popcount(counts[ptr]);
                            }
                        }
                        dfsRecalculateUppassLocal(
                            u->back, rootDownpass,
                            pr->partitionData[model], pr, w, tr->mxtips);
                    }
                }
                if (depth[nodesRecalculated.back().v->number] > 0) {
                    copySingleGlobalToLocalUppass(
                        pr,
                        uppass_par[nodesRecalculated.back().v->number]->number);
                }
                for (int i = (int)nodesRecalculated.size() - 1; i >= 0; --i) {
                    pair2 cur = nodesRecalculated[i];
                    nodeptr v = cur.v;
                    parsimonyNumber *vDownpass = cur.vDownpass;
                    parsimonyNumber *pUpVec;
//                    int stepPUp;
                    copySingleGlobalToLocalUppass(pr, v->number);
                    if (depth[v->number] == 0) {
                        assert(rootDownpass != NULL);
                        pUpVec = rootDownpass;
//                        stepPUp = INTS_PER_VECTOR;
                    } else {
                        nodeptr par = uppass_par[v->number];
                        assert(par != NULL);
//                        pUpVec = &(pr->partitionData[model]->parsVectUppassLocal
//                                       [width * states * par->number + w]);
                        pUpVec = &(pr->partitionData[model]->parsVectUppassLocal[init(par->number, w)]);
//                        stepPUp = width;
                    }
                    /* v MUST NOT be leaf */
                    assert(v->number > tr->mxtips);
                    nodeptr v1 = v->next->back;
                    nodeptr v2 = v->next->next->back;
                    parsimonyNumber *v1Downpass = NULL;
                    if (i > 0) {
                        nodeptr vTem = nodesRecalculated[i - 1].v;
                        if (v1 == vTem) {
                            v1Downpass = nodesRecalculated[i - 1].vDownpass;
                        } else {
                            swap(v1, v2);
                            assert(v1 == vTem);
                            v1Downpass = nodesRecalculated[i - 1].vDownpass;
                        }

                        assert(v1Downpass != NULL);
                        uppassInnerNodeCalculate(
                            vDownpass, v1Downpass,
//                            &(pr->partitionData[model]
//                                  ->parsVect[width * states * v2->number + w]),
                            &(pr->partitionData[model]
                                    ->parsVect[init(v2->number, w)]),
//                            &(pr->partitionData[model]->parsVectUppassLocal
//                                  [width * states * v->number + w]),
                            &(pr->partitionData[model]
                                ->parsVectUppassLocal[init(v->number, w)]),
                            pUpVec, states);
                    } else {
                        uppassInnerNodeCalculate(
                            vDownpass,
//                            &(pr->partitionData[model]
//                                  ->parsVect[width * states * v1->number + w]),
                            &(pr->partitionData[model]
                                    ->parsVect[init(v1->number, w)]),
//                            &(pr->partitionData[model]
//                                  ->parsVect[width * states * v2->number + w]),
                            &(pr->partitionData[model]
                                    ->parsVect[init(v2->number, w)]),
//                            &(pr->partitionData[model]
//                                ->parsVectUppassLocal[width * states * v->number + w]),
                            &(pr->partitionData[model]
                                    ->parsVectUppassLocal[init(v->number, w)]),
                            pUpVec, states);
                    }
                    // delete[] vDownpass;
                    if (vDownpass != NULL) {
                        rax_free(vDownpass);
                        vDownpass = NULL;
                    }

                    if (!equalStatesCmp(
//                            &(pr->partitionData[model]->parsVectUppassLocal
//                                  [width * states * v->number + w]),
                            &(pr->partitionData[model]
                                    ->parsVectUppassLocal[init(v->number, w)]),
//                            &(pr->partitionData[model]
//                                  ->parsVectUppass[width * states * v->number + w]),
                            &(pr->partitionData[model]
                                    ->parsVectUppass[init(v->number, w)]), states)) {
                        if (i > 0) {
                            dfsRecalculateUppassLocal(
                                v2,
//                                &(pr->partitionData[model]->parsVectUppassLocal
//                                      [width * states * v->number + w]),
                                &(pr->partitionData[model]
                                    ->parsVectUppassLocal[init(v->number, w)]),
                                pr->partitionData[model], pr, w, tr->mxtips);
                        } else {
                            if (v1 != Ns) {
                                assert(v2 == Ns);
                                dfsRecalculateUppassLocal(
                                    v1,
//                                    &(pr->partitionData[model]
//                                          ->parsVectUppassLocal[width * states * v->number + w]),
                                    &(pr->partitionData[model]
                                            ->parsVectUppassLocal[init(v->number, w)]),
                                    pr->partitionData[model], pr, w, tr->mxtips);
                            } else {
                                assert(v2 != Ns);
                                dfsRecalculateUppassLocal(
                                    v2,
//                                    &(pr->partitionData[model]
//                                          ->parsVectUppassLocal[width * states * v->number + w]),
                                    &(pr->partitionData[model]
                                            ->parsVectUppassLocal[init(v->number, w)]),
                                    pr->partitionData[model], pr, w, tr->mxtips);
                            }
                        }
                    }
                }
                if (rootDownpass != NULL) {
                    assert(depth[u->number] == 0);
                    // delete[] rootDownpass;
                    rax_free(rootDownpass);
                    rootDownpass = NULL;
                }
            }
            /* STEP 2c: Because parent of N_s is now N_z and not N_x anymore
             */
            copySingleGlobalToLocalUppass(pr, Nz->number);
            if (!equalStatesCmp(
                    &(pr->partitionData[model]->parsVectUppass[posNx]),
                    &(pr->partitionData[model]->parsVectUppassLocal[posNz]), states)) {
                dfsRecalculateUppassLocal(
                    Ns, &(pr->partitionData[model]->parsVectUppassLocal[posNz]),
                    pr->partitionData[model], pr, w, tr->mxtips);
            }
        }
    }
    // cout << "End recalculateDownpassAndUppass\n";
    return scoreMainSubtree + scoreClippedSubtree;
}
#else
#endif
void markNodesInRadiusRange(nodeptr p, int maxTips, int maxtrav) {
    inRadiusRange[p->number] = true;
    if (p->number <= maxTips || maxtrav <= 0) {
        return;
    }
    markNodesInRadiusRange(p->next->back, maxTips, maxtrav - 1);
    markNodesInRadiusRange(p->next->next->back, maxTips, maxtrav - 1);
}
void rearrangeTBR(pllInstance *tr, partitionList *pr, nodeptr p, int mintrav,
                  int maxtrav, int perSiteScores) {
    // cout << "Begin rearrangeTBR\n";
    /**
     * p: N_m
     * p1, p2: 2 children of p
     * q: N_x
     * q1, q2: N_z, N_s
     * Cut N_x from N_z and N_s
     * Connect N_z and N_s
     */
    nodeptr q = p->back;
    assert(depth[p->number] >= depth[q->number]);
    assert(p->number > tr->mxtips && q->number > tr->mxtips);
    nodeptr q1 = q->next->back;
    nodeptr q2 = q->next->next->back;
    nodeptr p1 = p->next->back;
    nodeptr p2 = p->next->next->back;
    /* Get the nodes in range [mintrav, maxtrav] needed for recalculation */
    for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
        inRadiusRange[i] = false;
    }
    markNodesInRadiusRange(p1, tr->mxtips, maxtrav);
    markNodesInRadiusRange(p2, tr->mxtips, maxtrav);
    markNodesInRadiusRange(q1, tr->mxtips, maxtrav);
    markNodesInRadiusRange(q2, tr->mxtips, maxtrav);
    q->next->back = q->next->next->back = NULL;
    q1->back = q2;
    q2->back = q1;
    /* Nz is node that has smaller depth (closer to the root) */
    /* Delay cut N_m (p) for later after recalculate downpass and uppass */
    scoreTwoSubtrees = recalculateDownpassAndUppass(tr, pr, p, q1, q2);

    /**
     * Cut N_m from 2 of its children
     * Connect 2 of its children together
     */
    p->next->back = p->next->next->back = NULL;
    p1->back = p2;
    p2->back = p1;
    // unsigned int correct = _evaluateParsimony(tr, pr, q1, PLL_TRUE, false) +
    //                        _evaluateParsimony(tr, pr, p1, PLL_TRUE, false);
    //
    // if (scoreTwoSubtrees != correct) {
    //     cout << "scoreTwoSubtrees = " << scoreTwoSubtrees << '\n';
    //     cout << "correct = " << correct << '\n';
    // }
    // assert(scoreTwoSubtrees == correct);
    // q->next->back = q1;
    // q->next->next->back = q2;
    // q1->back = q->next;
    // q2->back = q->next->next;
    // p->next->back = p1;
    // p->next->next->back = p2;
    // p1->back = p->next;
    // p2->back = p->next->next;
    // unsigned int tem = _evaluateParsimonyUppass(tr, pr, tr->start, false);
    // q->next->back = q->next->next->back = NULL;
    // q1->back = q2;
    // q2->back = q1;
    // p->next->back = p->next->next->back = NULL;
    // p1->back = p2;
    // p2->back = p1;
    assert(scoreTwoSubtrees <= oldScore);
    if (scoreTwoSubtrees < oldScore) {
        int count = 0, fi = count, se = 0;
        traversePrepareInsertBranches(pr, p1, maxtrav, 0, tr->mxtips, count);
        if (p2->number > tr->mxtips) {
            traversePrepareInsertBranches(pr, p2->next->back, maxtrav - 1, 1,
                                          tr->mxtips, count);
            traversePrepareInsertBranches(pr, p2->next->next->back, maxtrav - 1,
                                          1, tr->mxtips, count);
        }
        // cout << "count = " << count << '\n';
        se = count;
        traversePrepareInsertBranches(pr, q1, maxtrav, 0, tr->mxtips, count);
        if (q2->number > tr->mxtips) {
            traversePrepareInsertBranches(pr, q2->next->back, maxtrav - 1, 1,
                                          tr->mxtips, count);
            traversePrepareInsertBranches(pr, q2->next->next->back, maxtrav - 1,
                                          1, tr->mxtips, count);
        }
        // cout << "FI = " << fi << '\n';
        // cout << "SE = " << se << '\n';
        // cout << "count = " << count << '\n';
        for (int i = fi; i < se; ++i) {
            for (int j = se; j < count; ++j) {
                if (mintrav <= distFromRmvBranch[i] + distFromRmvBranch[j] &&
                    distFromRmvBranch[i] + distFromRmvBranch[j] <= maxtrav) {
                    cnt++;
                    unsigned int mp =
                        evaluateInsertParsimonyUppass(tr, pr, i, j);
                    // cout << "MP = " << mp << '\n';
                    if (mp < tr->bestParsimony) {
                        bestTreeScoreHits = 1;
                    } else if (mp == tr->bestParsimony) {
                        bestTreeScoreHits++;
                    }
                    if ((mp < tr->bestParsimony) ||
                        ((mp == tr->bestParsimony) &&
                         (random_double() <= 1.0 / bestTreeScoreHits))) {
                        tr->bestParsimony = mp;
                        tr->TBR_insertBranch1 = branchNode[i];
                        tr->TBR_insertBranch2 = branchNode[j];
                        tr->TBR_removeBranch = p;
                    }
                }
            }
        }
        // traverseInsertBranchesTBRP(tr, pr, p1, q1, p, perSiteScores);
        // if (q2->number > tr->mxtips) {
        //     traverseInsertBranchesTBRP(tr, pr, p1, q2->next->back, p,
        //                                perSiteScores);
        //     traverseInsertBranchesTBRP(tr, pr, p1, q2->next->next->back, p,
        //                                perSiteScores);
        // }
        // if (p2->number > tr->mxtips) {
        //     traverseInsertBranchesTBRP(tr, pr, p2->next->back, q1, p,
        //                                perSiteScores);
        //     traverseInsertBranchesTBRP(tr, pr, p2->next->next->back, q1, p,
        //                                perSiteScores);
        //     if (q2->number > tr->mxtips) {
        //         traverseInsertBranchesTBRP(tr, pr, p2->next->back,
        //                                    q2->next->back, p, perSiteScores);
        //         traverseInsertBranchesTBRP(tr, pr, p2->next->back,
        //                                    q2->next->next->back, p,
        //                                    perSiteScores);
        //         traverseInsertBranchesTBRP(tr, pr, p2->next->next->back,
        //                                    q2->next->back, p, perSiteScores);
        //         traverseInsertBranchesTBRP(tr, pr, p2->next->next->back,
        //                                    q2->next->next->back, p,
        //                                    perSiteScores);
        //     }
        // }
    }
    /* Reset isUppassCopied[] array to false */
    for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
        isUppassCopied[i] = false;
    }
    /**
     * Rollback to old tree
     * Reconnect N_x to N_s and N_z
     * Reconnect N_m to 2 of its children
     */
    q->next->back = q1;
    q->next->next->back = q2;
    q1->back = q->next;
    q2->back = q->next->next;
    p->next->back = p1;
    p->next->next->back = p2;
    p1->back = p->next;
    p2->back = p->next->next;
    // cout << "End rearrangeTBR\n";
}
void rearrangeSPR(pllInstance *tr, partitionList *pr, nodeptr p, int mintrav,
                  int maxtrav, int perSiteScores) {
    // cout << "Begin rearrangeTBR\n";
    /**
     * p: N_m
     * p1, p2: 2 children of p
     * q: N_x
     * q1, q2: N_z, N_s
     * Cut N_x from N_z and N_s
     * Connect N_z and N_s
     */
    nodeptr q = p->back;
    /* Nx must not be leaf at all cost */
    if (q->number <= tr->mxtips) {
        swap(p, q);
    }
    assert(depth[p->number] >= depth[q->number]);
    bool pIsLeaf = (p->number <= tr->mxtips);
    nodeptr q1 = q->next->back;
    nodeptr q2 = q->next->next->back;
    nodeptr p1 = NULL, p2 = NULL;
    /* Get the nodes in range [mintrav, maxtrav] needed for recalculation */
    for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
        inRadiusRange[i] = false;
    }
    markNodesInRadiusRange(q1, tr->mxtips, maxtrav);
    markNodesInRadiusRange(q2, tr->mxtips, maxtrav);
    if (!pIsLeaf) {
        p1 = p->next->back;
        p2 = p->next->next->back;
        markNodesInRadiusRange(p1, tr->mxtips, maxtrav);
        markNodesInRadiusRange(p2, tr->mxtips, maxtrav);
    }
    q->next->back = q->next->next->back = NULL;
    q1->back = q2;
    q2->back = q1;
    /* Nz is node that has smaller depth (closer to the root) */
    /* Delay cut N_m (p) for later after recalculate downpass and uppass */
    scoreTwoSubtrees = recalculateDownpassAndUppass(tr, pr, p, q1, q2);

    if (!pIsLeaf) {
        /**
         * Cut N_m from 2 of its children
         * Connect 2 of its children together
         */
        p->next->back = p->next->next->back = NULL;
        p1->back = p2;
        p2->back = p1;
    }
    // unsigned int correct = _evaluateParsimony(tr, pr, q1, PLL_TRUE, false) +
    //                        _evaluateParsimony(tr, pr, p1, PLL_TRUE, false);
    //
    // if (scoreTwoSubtrees != correct) {
    //     cout << "scoreTwoSubtrees = " << scoreTwoSubtrees << '\n';
    //     cout << "correct = " << correct << '\n';
    // }
    // assert(scoreTwoSubtrees == correct);
    // q->next->back = q1;
    // q->next->next->back = q2;
    // q1->back = q->next;
    // q2->back = q->next->next;
    // p->next->back = p1;
    // p->next->next->back = p2;
    // p1->back = p->next;
    // p2->back = p->next->next;
    // unsigned int tem = _evaluateParsimonyUppass(tr, pr, tr->start, false);
    // q->next->back = q->next->next->back = NULL;
    // q1->back = q2;
    // q2->back = q1;
    // p->next->back = p->next->next->back = NULL;
    // p1->back = p2;
    // p2->back = p1;
    assert(scoreTwoSubtrees <= oldScore);
    if (scoreTwoSubtrees < oldScore) {
        int count = 0, fi = count, se = 0;
        if (pIsLeaf) {
            for (int model = 0; model < pr->numberOfPartitions; ++model) {
                size_t states = pr->partitionData[model]->states,
                       width = pr->partitionData[model]->parsimonyLength;
                for (int i = width * states * p->number,
                         ptr = count * width * states;
                     i < width * states * (p->number + 1); ++i, ++ptr) {
                    pr->partitionData[model]->branchVectUppass[ptr] =
                        pr->partitionData[model]->parsVect[i];
                }
            }
            branchNode[count] = p;
            distFromRmvBranch[count] = 0;
            ++count;
        } else {
            traversePrepareInsertBranches(pr, p1, maxtrav, 0, tr->mxtips,
                                          count);
            if (p2->number > tr->mxtips) {
                traversePrepareInsertBranches(pr, p2->next->back, maxtrav - 1,
                                              1, tr->mxtips, count);
                traversePrepareInsertBranches(pr, p2->next->next->back,
                                              maxtrav - 1, 1, tr->mxtips,
                                              count);
            }
        }
        // cout << "count = " << count << '\n';
        se = count;
        traversePrepareInsertBranches(pr, q1, maxtrav, 0, tr->mxtips, count);
        if (q2->number > tr->mxtips) {
            traversePrepareInsertBranches(pr, q2->next->back, maxtrav - 1, 1,
                                          tr->mxtips, count);
            traversePrepareInsertBranches(pr, q2->next->next->back, maxtrav - 1,
                                          1, tr->mxtips, count);
        }
        // cout << "FI = " << fi << '\n';
        // cout << "SE = " << se << '\n';
        // cout << "count = " << count << '\n';
        for (int i = fi; i < se; ++i) {
            for (int j = se; j < count; ++j) {
                if ((i == fi || j == se) &&
                    (mintrav <= distFromRmvBranch[i] + distFromRmvBranch[j] &&
                     distFromRmvBranch[i] + distFromRmvBranch[j] <= maxtrav)) {
                    cnt++;
                    unsigned int mp =
                        evaluateInsertParsimonyUppass(tr, pr, i, j);
                    // cout << "MP = " << mp << '\n';
                    if (mp < tr->bestParsimony) {
                        bestTreeScoreHits = 1;
                    } else if (mp == tr->bestParsimony) {
                        bestTreeScoreHits++;
                    }
                    if ((mp < tr->bestParsimony) ||
                        ((mp == tr->bestParsimony) &&
                         (random_double() <= 1.0 / bestTreeScoreHits))) {
                        tr->bestParsimony = mp;
                        /* TBR_removeBranch MUST saved the node that is on
                         * different side against TBR_insertBranch1 among 2
                         * nodes p-q of remove branch */
                        if (i == fi) {
                            tr->TBR_insertBranch1 = branchNode[j];
                            tr->TBR_removeBranch = q;
                        } else {
                            tr->TBR_insertBranch1 = branchNode[i];
                            tr->TBR_removeBranch = p;
                        }
                        tr->TBR_insertBranch2 = NULL;
                    }
                }
            }
        }
        // traverseInsertBranchesTBRP(tr, pr, p1, q1, p, perSiteScores);
        // if (q2->number > tr->mxtips) {
        //     traverseInsertBranchesTBRP(tr, pr, p1, q2->next->back, p,
        //                                perSiteScores);
        //     traverseInsertBranchesTBRP(tr, pr, p1, q2->next->next->back, p,
        //                                perSiteScores);
        // }
        // if (p2->number > tr->mxtips) {
        //     traverseInsertBranchesTBRP(tr, pr, p2->next->back, q1, p,
        //                                perSiteScores);
        //     traverseInsertBranchesTBRP(tr, pr, p2->next->next->back, q1, p,
        //                                perSiteScores);
        //     if (q2->number > tr->mxtips) {
        //         traverseInsertBranchesTBRP(tr, pr, p2->next->back,
        //                                    q2->next->back, p, perSiteScores);
        //         traverseInsertBranchesTBRP(tr, pr, p2->next->back,
        //                                    q2->next->next->back, p,
        //                                    perSiteScores);
        //         traverseInsertBranchesTBRP(tr, pr, p2->next->next->back,
        //                                    q2->next->back, p, perSiteScores);
        //         traverseInsertBranchesTBRP(tr, pr, p2->next->next->back,
        //                                    q2->next->next->back, p,
        //                                    perSiteScores);
        //     }
        // }
    }
    /* Reset isUppassCopied[] array to false */
    for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
        isUppassCopied[i] = false;
    }
    /**
     * Rollback to old tree
     * Reconnect N_x to N_s and N_z
     * Reconnect N_m to 2 of its children
     */
    q->next->back = q1;
    q->next->next->back = q2;
    q1->back = q->next;
    q2->back = q->next->next;
    if (!pIsLeaf) {
        p->next->back = p1;
        p->next->next->back = p2;
        p1->back = p->next;
        p2->back = p->next->next;
    }
    // cout << "End rearrangeTBR\n";
}
void applySPRTreeStructure(nodeptr p, nodeptr i1) {
    /*
     * p is the remove branch
     * i1 is the insert branch
     */
    nodeptr p1 = p->next->back;
    nodeptr p2 = p->next->next->back;
    p1->back = p2;
    p2->back = p1;

    nodeptr i2 = i1->back;
    p->next->back = i1;
    p->next->next->back = i2;
    i1->back = p->next;
    i2->back = p->next->next;
}
void applyTBRTreeStructure(nodeptr u, nodeptr p1, nodeptr q1) {
    /*
     * u is remove branch
     * p1, q1 are 2 insert branches
     */
    nodeptr v = u->back;
    nodeptr u1 = u->next->back;
    nodeptr u2 = u->next->next->back;
    nodeptr v1 = v->next->back;
    nodeptr v2 = v->next->next->back;
    // cout << "Remove branch = " << u->number << ' ' << v->number << '\n';
    // cout << "Insert branch 1 = " << p1->number << ' ' << p1->back->number
    //      << '\n';
    // cout << "Insert branch 2 = " << q1->number << ' ' << q1->back->number
    //      << '\n';
    u1->back = u2;
    u2->back = u1;
    v1->back = v2;
    v2->back = v1;

    nodeptr p2 = p1->back;
    nodeptr q2 = q1->back;
    p1->back = u->next;
    p2->back = u->next->next;
    q1->back = v->next;
    q2->back = v->next->next;
    u->next->back = p1;
    u->next->next->back = p2;
    v->next->back = q1;
    v->next->next->back = q2;
}

static void markRecalculatedNode(nodeptr u) {
    /* WARNING: recalculate[root] and recalculate[root->back] MUST be true
     * before calling this function */
    while (recalculate[u->number] == false) {
        recalculate[u->number] = true;
        u = uppass_par[u->number];
    }
}
static void applyMove(pllInstance *tr, partitionList *pr) {
    recalculate[tr->start->number] = recalculate[tr->start->back->number] =
        true;
    // cout << "Old root: " << tr->start->number << " - "
    //      << tr->start->back->number << '\n';
    nodeptr r = tr->TBR_removeBranch;
    nodeptr i1 = tr->TBR_insertBranch1;
    nodeptr i2 = tr->TBR_insertBranch2;
    if (i2 != NULL) {
        // TBR
        markRecalculatedNode(
            (depth[i1->number] < depth[i1->back->number] ? i1 : i1->back));
        markRecalculatedNode(
            (depth[i2->number] < depth[i2->back->number] ? i2 : i2->back));
        applyTBRTreeStructure(r, i1, i2);
    } else {
        // SPR
        markRecalculatedNode(
            (depth[i1->number] < depth[i1->back->number] ? i1 : i1->back));
        markRecalculatedNode(
            (depth[r->number] < depth[r->back->number] ? r : r->back));
        applySPRTreeStructure(r, i1);
    }
    recalculate[tr->start->number] = recalculate[tr->start->back->number] =
        true;
    for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
        if (recalculate[i]) {
            copySingleGlobalToLocalUppass(pr, i);
        }
    }
    // copyGlobalToLocalUppass(pr, 2 * tr->mxtips - 1);
    int *ti = tr->ti, counter = 4;

    ti[1] = tr->start->number;
    ti[2] = tr->start->back->number;
    // cout << "pNumber = " << p->number << '\n';
    // cout << "qNumber = " << q->number << '\n';
    // cout << "New root: " << tr->start->number << " - "
    //      << tr->start->back->number << '\n';
    computeTraversalInfoParsimonyUppass(tr->start, ti, &counter, tr->mxtips,
                                        false, true);
    computeTraversalInfoParsimonyUppass(tr->start->back, ti, &counter,
                                        tr->mxtips, false, true);

    ti[0] = counter;
    oldScore = _evaluateParsimonyIterativeFastUppass(tr, pr, false);

    for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
        nodeptr u = tr->nodep[i];
        if (recalculate[u->number]) {
            int numPtrs = (u->number <= tr->mxtips ? 1 : 3);
            for (int j = 0; j < numPtrs; ++j, u = u->next) {
                nodeptr v = u->back;
                if (!recalculate[v->number]) {
                    for (int model = 0; model < pr->numberOfPartitions;
                         ++model) {
                        pInfo *prModel = pr->partitionData[model];
                        int states = pr->partitionData[model]->states,
                            width = pr->partitionData[model]->parsimonyLength;
                        auto init = [width, states](std::size_t node_id, std::size_t col_id = 0) -> std::size_t {
                            return table_index(width, states, node_id, 0, col_id);
                        };

                        for (int w = 0; w < width; w += INTS_PER_VECTOR) {
                            /**
                             * NOTE: u                 is new parent of v
                             *       uppass[v->number] is old parent of v
                             */
                            if (!equalStatesCmp(
//                                    &(prModel->parsVectUppassLocal
//                                          [width * states * uppass_par[v->number]->number + w]),
                                    &(prModel
                                        ->parsVectUppassLocal[init(uppass_par[v->number]->number, w)]),
//                                    &(prModel->parsVectUppass
//                                          [width * states * u->number + w]),
                                    &(prModel
                                        ->parsVectUppass[init(u->number, w)]), states)) {
                                dfsRecalculateUppass(
                                    v,
//                                    &(prModel->parsVectUppass
//                                          [width * states * u->number + w]),
                                    &(prModel->parsVectUppass[init(u->number, w)]),
                                    prModel, pr, w, tr->mxtips);
                            }
                        }
                    }
                }
            }
        }
    }
    /* Reset recalculate[] array to false */
    for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
        recalculate[i] = isUppassCopied[i] = false;
    }
}
static void reorderNodesUppass(pllInstance *tr, nodeptr *np, nodeptr p,
                               int *count) {
    if ((p->number <= tr->mxtips))
        return;
    else {
        tr->nodep[*count + tr->mxtips + 1] = p;
        ++(*count);

        nodeptr q = p->next->back;
        nodeptr r = p->next->next->back;
        uppass_par[q->number] = p;
        uppass_par[r->number] = p;
        depth[q->number] = depth[r->number] = depth[p->number] + 1;
        reorderNodesUppass(tr, np, q, count);
        reorderNodesUppass(tr, np, r, count);
    }
}

static void nodeRectifierParsUppass(pllInstance *tr) {
    nodeptr *np = (nodeptr *)rax_malloc(2 * tr->mxtips * sizeof(nodeptr));
    int count = 0;

    // tr->start = tr->nodep[1];
    // tr->rooted = PLL_FALSE;
    //
    // /* TODO why is tr->rooted set to PLL_FALSE here ?*/

    for (int i = tr->mxtips + 1; i <= (tr->mxtips + tr->mxtips - 1); ++i)
        np[i] = tr->nodep[i];
    uppass_par[tr->start->number] = uppass_par[tr->start->back->number] = NULL;
    depth[tr->start->number] = depth[tr->start->back->number] = 0;
    reorderNodesUppass(tr, np, tr->start, &count);
    reorderNodesUppass(tr, np, tr->start->back, &count);

    rax_free(np);
}
/**
 * Return number of leaves in subtree u
 * Find the middle branch of the tree, assigned to tr->start
 */
int getNumLeavesSubtree(pllInstance *tr, nodeptr u) {
    // cout << "Edge: " << u->number << ' ' << u->back->number << '\n';
    if (u->number <= tr->mxtips) {
        return 1;
    }
    int num = getNumLeavesSubtree(tr, u->next->back) +
              getNumLeavesSubtree(tr, u->next->next->back);
    if (tr->start == NULL && num * 2 >= tr->mxtips) {
        tr->start = u;
    }
    return num;
}
int pllOptimizeTbrUppass(pllInstance *tr, partitionList *pr, int mintrav,
                         int maxtrav, IQTree *_iqtree) {
    // cout << "Begin pllOptimizeTbrUppass\n";
    int perSiteScores = globalParam->gbo_replicates > 0;

    iqtree = _iqtree; // update pointer to IQTree

    if (globalParam->ratchet_iter >= 0 &&
        (iqtree->on_ratchet_hclimb1 || iqtree->on_ratchet_hclimb2)) {
        // oct 23: in non-ratchet iteration, allocate is not triggered
        _updateInternalPllOnRatchet(tr, pr);
        _allocateParsimonyDataStructuresUppass(
            tr, pr, perSiteScores); // called once if not running ratchet
    } else if (first_call || (iqtree && iqtree->on_opt_btree))
        _allocateParsimonyDataStructuresUppass(
            tr, pr, perSiteScores); // called once if not running ratchet

    if (first_call) {
        first_call = false;
    }
    unsigned int randomMP, startMP;
    assert(!tr->constrained);

    tr->start = NULL;
    getNumLeavesSubtree(tr, tr->nodep[1]->back);
    // cout << "Root: " << tr->start->number << ' ' << tr->start->back->number
    //      << '\n';
    // nodeRectifierPars(tr);
    tr->bestParsimony =
        _evaluateParsimonyUppass(tr, pr, tr->start, perSiteScores);
    // cout << "tr->bestParsimony = " << tr->bestParsimony << '\n';

    assert(-iqtree->curScore == tr->bestParsimony);
    // cout << "Start MP = " << tr->bestParsimony << '\n';
    unsigned int bestIterationScoreHits = 1;
    randomMP = tr->bestParsimony;
    oldScore = tr->bestParsimony;
    tr->TBR_removeBranch = NULL;
    int lastNodepId = -1;
    nodeRectifierParsUppass(tr);
    do {
        startMP = randomMP;
        /* Iterate through all remove-branches */
        for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
            // cout << "Remove branch: " << i << ' ' << tr->nodep[i]->number <<
            // ' '
            //      << tr->nodep[i]->back->number << '\n';
            if (lastNodepId != -1 && i == lastNodepId) {
                // cout << "Don't consider last remove-branch\n";
                break;
            }
            if (i > tr->mxtips && tr->nodep[i]->back->number > tr->mxtips) {
                rearrangeTBR(tr, pr, tr->nodep[i], mintrav, maxtrav,
                             perSiteScores);
            } else {
                rearrangeSPR(tr, pr, tr->nodep[i], mintrav, maxtrav,
                             perSiteScores);
            }
            if (tr->bestParsimony == randomMP)
                bestIterationScoreHits++;
            if (tr->bestParsimony < randomMP) {
                bestIterationScoreHits = 1;
            }
            if (((tr->bestParsimony < randomMP) ||
                 ((tr->bestParsimony == randomMP) &&
                  (random_double() <= 1.0 / bestIterationScoreHits))) &&
                tr->TBR_removeBranch && tr->TBR_insertBranch1) {
                // cout << "remove branch: " << tr->TBR_removeBranch->number
                //      << " - " << tr->TBR_removeBranch->back->number << '\n';
                // cout << "insert branch: " << tr->TBR_insertBranch1->number
                //      << " - " << tr->TBR_insertBranch1->back->number << '\n';
                applyMove(tr, pr);
                randomMP = oldScore;
                // int temMP = _evaluateParsimony(tr, pr, tr->start, true,
                // false); randomMP =
                //     _evaluateParsimonyUppass(tr, pr, tr->start,
                //     perSiteScores);
                // assert(randomMP == oldScore);
                // if (randomMP != tr->bestParsimony) {
                //     cout << "remove branch: " << tr->TBR_removeBranch->number
                //          << " - " << tr->TBR_removeBranch->back->number << '\n';
                //     cout << "insert branch: " << tr->TBR_insertBranch1->number
                //          << " - " << tr->TBR_insertBranch1->back->number
                //          << '\n';
                //     // cout << "insert branch: " <<
                //     // tr->TBR_insertBranch2->number
                //     //      << " - " << tr->TBR_insertBranch2->back->number
                //     // <<
                //     //      '\n';
                //     cout << "randomMP = " << randomMP << '\n';
                //     cout << "tr->bestParsimony = " << tr->bestParsimony << '\n';
                //     randomMP = _evaluateParsimonyUppass(tr, pr, tr->start,
                //                                         perSiteScores);
                //     cout << "Correct randomMP = " << randomMP << '\n';
                //     assert(0);
                // }
                lastNodepId = i;
                tr->TBR_removeBranch = NULL;
                tr->TBR_insertBranch1 = tr->TBR_insertBranch2 = NULL;
                bestTreeScoreHits = 1;
                assert(randomMP == tr->bestParsimony);
                nodeRectifierParsUppass(tr);
            }
        }
    } while (randomMP < startMP);
    // cout << "CNT = " << cnt << '\n';
    tr->start = tr->nodep[1];
    // cout << "End pllOptimizeTbrUppass\n";
    return startMP;
}
int pllOptimizeSprUppass(pllInstance *tr, partitionList *pr, int mintrav,
                         int maxtrav, IQTree *_iqtree) {
    // cout << "Begin pllOptimizeTbrUppass\n";
    int perSiteScores = globalParam->gbo_replicates > 0;

    iqtree = _iqtree; // update pointer to IQTree

    if (globalParam->ratchet_iter >= 0 &&
        (iqtree->on_ratchet_hclimb1 || iqtree->on_ratchet_hclimb2)) {
        // oct 23: in non-ratchet iteration, allocate is not triggered
        _updateInternalPllOnRatchet(tr, pr);
        _allocateParsimonyDataStructuresUppass(
            tr, pr, perSiteScores); // called once if not running ratchet
    } else if (first_call || (iqtree && iqtree->on_opt_btree))
        _allocateParsimonyDataStructuresUppass(
            tr, pr, perSiteScores); // called once if not running ratchet

    if (first_call) {
        first_call = false;
    }
    unsigned int randomMP, startMP;
    assert(!tr->constrained);

    tr->start = NULL;
    getNumLeavesSubtree(tr, tr->nodep[1]->back);
    // cout << "Root: " << tr->start->number << ' ' << tr->start->back->number
    //      << '\n';
    // nodeRectifierPars(tr);
    tr->bestParsimony =
        _evaluateParsimonyUppass(tr, pr, tr->start, perSiteScores);
    // cout << "tr->bestParsimony = " << tr->bestParsimony << '\n';

    assert(-iqtree->curScore == tr->bestParsimony);
    // cout << "Start MP = " << tr->bestParsimony << '\n';
    unsigned int bestIterationScoreHits = 1;
    randomMP = tr->bestParsimony;
    oldScore = tr->bestParsimony;
    tr->TBR_removeBranch = NULL;
    int lastNodepId = -1;
    nodeRectifierParsUppass(tr);
    do {
        startMP = randomMP;
        /* Iterate through all remove-branches */
        for (int i = 1; i <= tr->mxtips + tr->mxtips - 2; ++i) {
            // cout << "Remove branch: " << i << ' ' << tr->nodep[i]->number <<
            // ' '
            //      << tr->nodep[i]->back->number << '\n';
            if (lastNodepId != -1 && i == lastNodepId) {
                // cout << "Don't consider last remove-branch\n";
                break;
            }
            rearrangeSPR(tr, pr, tr->nodep[i], mintrav, maxtrav,
                         perSiteScores);
            if (tr->bestParsimony == randomMP)
                bestIterationScoreHits++;
            if (tr->bestParsimony < randomMP) {
                bestIterationScoreHits = 1;
            }
            if (((tr->bestParsimony < randomMP) ||
                 ((tr->bestParsimony == randomMP) &&
                  (random_double() <= 1.0 / bestIterationScoreHits))) &&
                tr->TBR_removeBranch && tr->TBR_insertBranch1) {
                // cout << "remove branch: " << tr->TBR_removeBranch->number
                //      << " - " << tr->TBR_removeBranch->back->number << '\n';
                // cout << "insert branch: " << tr->TBR_insertBranch1->number
                //      << " - " << tr->TBR_insertBranch1->back->number << '\n';
                applyMove(tr, pr);
                randomMP = oldScore;
                // int temMP = _evaluateParsimony(tr, pr, tr->start, true,
                // false); randomMP =
                //     _evaluateParsimonyUppass(tr, pr, tr->start,
                //     perSiteScores);
                // assert(randomMP == oldScore);
                // if (randomMP != tr->bestParsimony) {
                //     cout << "remove branch: " << tr->TBR_removeBranch->number
                //          << " - " << tr->TBR_removeBranch->back->number <<
                //          '\n';
                //     cout << "insert branch: " <<
                //     tr->TBR_insertBranch1->number
                //          << " - " << tr->TBR_insertBranch1->back->number
                //          << '\n';
                //     // cout << "insert branch: " <<
                //     // tr->TBR_insertBranch2->number
                //     //      << " - " << tr->TBR_insertBranch2->back->number
                //     // <<
                //     //      '\n';
                //     cout << "randomMP = " << randomMP << '\n';
                //     cout << "tr->bestParsimony = " << tr->bestParsimony <<
                //     '\n'; randomMP = _evaluateParsimonyUppass(tr, pr,
                //     tr->start,
                //                                         perSiteScores);
                //     cout << "Correct randomMP = " << randomMP << '\n';
                //     assert(0);
                // }
                lastNodepId = i;
                tr->TBR_removeBranch = NULL;
                tr->TBR_insertBranch1 = tr->TBR_insertBranch2 = NULL;
                bestTreeScoreHits = 1;
                assert(randomMP == tr->bestParsimony);
                nodeRectifierParsUppass(tr);
            }
        }
    } while (randomMP < startMP);
    // cout << "CNT = " << cnt << '\n';
    tr->start = tr->nodep[1];
    // cout << "End pllOptimizeTbrUppass\n";
    return startMP;
}
