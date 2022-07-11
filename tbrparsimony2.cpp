/**
 * tbrparsimony.cpp
 * NOTE: Use functions the same as in sprparsimony.cpp, so I have to declare it
 * static (globally can't have functions or variables with the same name)
 */
#include <algorithm>
#include "sprparsimony2.h"
#include "tbrparsimony2.h"

#include "nnisearch.h"
#include "parstree.h"
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
//#define LONG_INTS_PER_VECTOR (64/sizeof(long))
#define INT_TYPE __m512i
#define CAST double *
#define SET_ALL_BITS_ONE _mm512_set1_epi32(0xFFFFFFFF)
#define SET_ALL_BITS_ZERO _mm512_setzero_epi32()
#define VECTOR_LOAD _mm512_load_epi32
#define VECTOR_STORE _mm512_store_epi32
#define VECTOR_BIT_AND _mm512_and_epi32
#define VECTOR_BIT_OR _mm512_or_epi32
#define VECTOR_AND_NOT _mm512_andnot_epi32

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
//#define LONG_INTS_PER_VECTOR (32/sizeof(long))
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
//#define LONG_INTS_PER_VECTOR (16/sizeof(long))
#else
#define ULINT_SIZE 64
#define LONG_INTS_PER_VECTOR 2
//#define LONG_INTS_PER_VECTOR (16/sizeof(long))
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

#else
// no vectorization
#define VECTOR_SIZE 1
#endif

#include "pllrepo/src/pll.h"
#include "pllrepo/src/pllInternal.h"

extern const unsigned int mask32[32];
// /* vector-specific stuff */

extern double masterTime;

// /* program options */
extern Params *globalParam;
static IQTree *iqtree = NULL;
static unsigned long bestTreeScoreHits; // to count hits to bestParsimony

extern parsimonyNumber *pllCostMatrix;    // Diep: For weighted version
extern int pllCostNstates;                // Diep: For weighted version
extern parsimonyNumber *vectorCostMatrix; // BQM: vectorized cost matrix
static parsimonyNumber highest_cost;

// //(if needed) split the parsimony vector into several segments to avoid
// overflow when calc rell based on vec8us
extern int pllRepsSegments;  // # of segments
extern int *pllSegmentUpper; // array of first index of the next segment, see
                             // IQTree::segment_upper
static parsimonyNumber
    *pllRemainderLowerBounds; // array of lower bound score for the
                              // un-calculated part to the right of a segment
static bool doing_stepwise_addition = false; // is the stepwise addition on
static bool first_call = true;
static int numDrawTrees = 0;
static string lastTreeString = "";
static void initializeCostMatrix() {
    highest_cost =
        *max_element(pllCostMatrix,
                     pllCostMatrix + pllCostNstates * pllCostNstates) +
        1;

    //    cout << "Segments: ";
    //    for (int i = 0; i < pllRepsSegments; i++)
    //        cout <<  " " << pllSegmentUpper[i];
    //    cout << endl;

#if (defined(__SSE3) || defined(__AVX))
    assert(pllCostMatrix);
    if (!vectorCostMatrix) {
        rax_posix_memalign((void **)&(vectorCostMatrix), PLL_BYTE_ALIGNMENT,
                           sizeof(parsimonyNumber) * pllCostNstates *
                               pllCostNstates);

        if (globalParam->sankoff_short_int) {
            parsimonyNumberShort *shortMatrix =
                (parsimonyNumberShort *)vectorCostMatrix;
            // duplicate the cost entries for vector operations
            for (int i = 0; i < pllCostNstates; i++)
                for (int j = 0; j < pllCostNstates; j++)
                    shortMatrix[(i * pllCostNstates + j)] =
                        pllCostMatrix[i * pllCostNstates + j];
        } else {
            // duplicate the cost entries for vector operations
            for (int i = 0; i < pllCostNstates; i++)
                for (int j = 0; j < pllCostNstates; j++)
                    vectorCostMatrix[(i * pllCostNstates + j)] =
                        pllCostMatrix[i * pllCostNstates + j];
        }
    }
#else
    vectorCostMatrix = NULL;
#endif
}

// note: pllCostMatrix[i*pllCostNstates+j] = cost from i to j

///************************************************ pop count stuff
///***********************************************/
//
// unsigned int bitcount_32_bit(unsigned int i)
//{
//  return ((unsigned int) __builtin_popcount(i));
//}

///* bit count for 64 bit integers */
//
// inline unsigned int bitcount_64_bit(unsigned long i)
//{
//  return ((unsigned int) __builtin_popcountl(i));
//}

/* bit count for 128 bit SSE3 and 256 bit AVX registers */

#if (defined(__SSE3) || defined(__AVX))
static inline unsigned int vectorPopcount(INT_TYPE v) {
    unsigned long counts[LONG_INTS_PER_VECTOR]
        __attribute__((aligned(PLL_BYTE_ALIGNMENT)));

    int i, sum = 0;

    VECTOR_STORE((CAST)counts, v);

    for (i = 0; i < LONG_INTS_PER_VECTOR; i++)
        sum += __builtin_popcountl(counts[i]);

    return ((unsigned int)sum);
}
#endif

/********************************DNA FUNCTIONS
 * *****************************************************************/

// Diep:
// store per site score to nodeNumber
#if (defined(__SSE3) || defined(__AVX))
static inline void storePerSiteNodeScores(partitionList *pr, int model,
                                          INT_TYPE v, unsigned int offset,
                                          int nodeNumber) {

    unsigned long counts[LONG_INTS_PER_VECTOR]
        __attribute__((aligned(PLL_BYTE_ALIGNMENT)));
    parsimonyNumber *buf;

    int i, j;

    VECTOR_STORE((CAST)counts, v);

    int partialParsLength = pr->partitionData[model]->parsimonyLength * PLL_PCF;
    int nodeStart = partialParsLength * nodeNumber;
    int nodeStartPlusOffset = nodeStart + offset * PLL_PCF;
    for (i = 0; i < LONG_INTS_PER_VECTOR; ++i) {
        buf = &(
            pr->partitionData[model]->perSitePartialPars[nodeStartPlusOffset]);
        nodeStartPlusOffset += ULINT_SIZE;
        //		buf =
        //&(pr->partitionData[model]->perSitePartialPars[nodeStart +
        // offset * PLL_PCF + i * ULINT_SIZE]); // Diep's 		buf =
        //&(pr->partitionData[model]->perSitePartialPars[nodeStart + offset *
        // PLL_PCF + i]); // Tomas's code
        for (j = 0; j < ULINT_SIZE; ++j)
            buf[j] += ((counts[i] >> j) & 1);
    }
}

// Diep:
// Add site scores in q and r to p
// q and r are children of p
template <class VectorClass>
void addPerSiteSubtreeScoresSIMD(partitionList *pr, int pNumber, int qNumber,
                                 int rNumber) {
    assert(VectorClass::size() == INTS_PER_VECTOR);
    parsimonyNumber *pBuf, *qBuf, *rBuf;
    for (int i = 0; i < pr->numberOfPartitions; i++) {
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
    //	for(int i = 0; i < pr->numberOfPartitions; i++){
    //		int partialParsLength = pr->partitionData[i]->parsimonyLength *
    // PLL_PCF; 		pBuf =
    // &(pr->partitionData[i]->perSitePartialPars[partialParsLength
    //* pNumber]); 		qBuf =
    //&(pr->partitionData[i]->perSitePartialPars[partialParsLength * qNumber]);
    //		rBuf =
    //&(pr->partitionData[i]->perSitePartialPars[partialParsLength
    //* rNumber]); 		for(int k = 0; k < partialParsLength; k++)
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
    for (int i = 0; i < pr->numberOfPartitions; i++) {
        int partialParsLength = pr->partitionData[i]->parsimonyLength * PLL_PCF;
        pBuf = &(pr->partitionData[i]
                     ->perSitePartialPars[partialParsLength * pNumber]);
        memset(pBuf, 0, partialParsLength * sizeof(parsimonyNumber));
    }
}
#endif

static int checkerPars(pllInstance *tr, nodeptr p) {
    int group = tr->constraintVector[p->number];

    if (isTip(p->number, tr->mxtips)) {
        group = tr->constraintVector[p->number];
        return group;
    } else {
        if (group != -9)
            return group;

        group = checkerPars(tr, p->next->back);
        if (group != -9)
            return group;

        group = checkerPars(tr, p->next->next->back);
        if (group != -9)
            return group;

        return -9;
    }
}

static pllBoolean tipHomogeneityCheckerPars(pllInstance *tr, nodeptr p,
                                            int grouping) {
    if (isTip(p->number, tr->mxtips)) {
        if (tr->constraintVector[p->number] != grouping)
            return PLL_FALSE;
        else
            return PLL_TRUE;
    } else {
        return (tipHomogeneityCheckerPars(tr, p->next->back, grouping) &&
                tipHomogeneityCheckerPars(tr, p->next->next->back, grouping));
    }
}

static void getxnodeLocal(nodeptr p) {
    nodeptr s;

    if ((s = p->next)->xPars || (s = s->next)->xPars) {
        p->xPars = s->xPars;
        s->xPars = 0;
    }

    assert(p->next->xPars || p->next->next->xPars || p->xPars);
}
static bool needRecalculate(nodeptr p, int maxTips) {
    if (p->number <= maxTips)
        return p->recalculate;
    return p->recalculate || p->next->recalculate || p->next->next->recalculate;
}
template <int perSiteScores>
static void computeTraversalInfoParsimonyTBR(nodeptr p, int *ti, int *counter,
                                             int maxTips) {
#if (defined(__SSE3) || defined(__AVX))
    if (perSiteScores && pllCostMatrix == NULL) {
        resetPerSiteNodeScores(iqtree->pllPartitions, p->number);
    }
#endif
    p->recalculate = false;
    if (p->number <= maxTips)
        return;
    if (!p->xPars)
        getxnodeLocal(p);
    p->next->recalculate = p->next->next->recalculate = false;
    nodeptr q = p->next->back, r = p->next->next->back;
    // cout << "computeTraversal\n";
    q->par = r->par = p;
    if (needRecalculate(q, maxTips))
        computeTraversalInfoParsimonyTBR<perSiteScores>(q, ti, counter, maxTips);

    if (needRecalculate(r, maxTips))
        computeTraversalInfoParsimonyTBR<perSiteScores>(r, ti, counter, maxTips);

    ti[*counter] = p->number;
    ti[*counter + 1] = q->number;
    ti[*counter + 2] = r->number;
    *counter = *counter + 4;
}

template <int perSiteScores>
static void computeTraversalInfoParsimony(nodeptr p, int *ti, int *counter,
                                          int maxTips, pllBoolean full) {
#if (defined(__SSE3) || defined(__AVX))
    if (perSiteScores && pllCostMatrix == NULL) {
        resetPerSiteNodeScores(iqtree->pllPartitions, p->number);
    }
#endif

    nodeptr q = p->next->back, r = p->next->next->back;

    if (!p->xPars)
        getxnodeLocal(p);

    if (full) {
        if (q->number > maxTips)
            computeTraversalInfoParsimony<perSiteScores>(q, ti, counter, maxTips, full);

        if (r->number > maxTips)
            computeTraversalInfoParsimony<perSiteScores>(r, ti, counter, maxTips, full);
    } else {
        if (q->number > maxTips && !q->xPars)
            computeTraversalInfoParsimony<perSiteScores>(q, ti, counter, maxTips, full);

        if (r->number > maxTips && !r->xPars)
            computeTraversalInfoParsimony<perSiteScores>(r, ti, counter, maxTips, full);
    }

    ti[*counter] = p->number;
    ti[*counter + 1] = q->number;
    ti[*counter + 2] = r->number;
    *counter = *counter + 4;
}

static void getRecalculateNodeTBR(nodeptr root, nodeptr root1, nodeptr u,
                                  nodeptr v) {
    root->recalculate = root1->recalculate = true;
    while (u != root && u != root1 && u->recalculate == false) {
        // assert(u != NULL);
        u->recalculate = true;
        u = u->par;
    }
    while (v != root && v != root1 && v->recalculate == false) {
        // assert(v != NULL);
        v->recalculate = true;
        v = v->par;
    }
}

template <int perSiteScores>
static unsigned int evaluateParsimonyTBR(pllInstance *tr, partitionList *pr,
                                         nodeptr u, nodeptr v, nodeptr w) {
    volatile unsigned int result;
    nodeptr p = tr->curRoot;
    nodeptr q = tr->curRootBack;
    int *ti = tr->ti, counter = 4;

    ti[1] = w->number;
    ti[2] = w->back->number;
    getRecalculateNodeTBR(p, q, u, v);
    computeTraversalInfoParsimonyTBR<perSiteScores>(w, ti, &counter, tr->mxtips);
    computeTraversalInfoParsimonyTBR<perSiteScores>(w->back, ti, &counter, tr->mxtips);
    ti[0] = counter;
    result = _evaluateParsimonyIterativeFast<perSiteScores>(tr, pr);
    return result;
}

/****************************************************************************************************************************************/
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

    for (model = 0; model < pr->numberOfPartitions; model++) {

        for (i = pr->partitionData[model]->lower;
             i < pr->partitionData[model]->upper; i++) {
            if (isInformative(tr, pr->partitionData[model]->dataType, i)) {
                informative[i] = 1;
            } else {
                informative[i] = 0;
            }
        }
    }

    /* printf("Uninformative Patterns: %d\n", number); */
}
template <class Numeric, const int VECSIZE>
static void compressSankoffDNA(pllInstance *tr, partitionList *pr,
                               int *informative, int perSiteScores) {
    //	cout << "Begin compressSankoffDNA()" << endl;
    size_t totalNodes, i, model;

    totalNodes = 2 * (size_t)tr->mxtips;

    for (model = 0; model < (size_t)pr->numberOfPartitions; model++) {
        size_t k, states = (size_t)pr->partitionData[model]->states,
                  compressedEntries, compressedEntriesPadded, entries = 0,
                  lower = pr->partitionData[model]->lower,
                  upper = pr->partitionData[model]->upper;

        //      parsimonyNumber
        //        **compressedTips = (parsimonyNumber **)rax_malloc(states *
        //        sizeof(parsimonyNumber*)), *compressedValues =
        //        (parsimonyNumber
        //        *)rax_malloc(states * sizeof(parsimonyNumber));

        for (i = lower; i < upper; i++)
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
        for (i = 0; i < (size_t)tr->mxtips; i++) {
            size_t w = 0, compressedIndex = 0, compressedCounter = 0, index = 0,
                   informativeIndex = 0;

            //          for(k = 0; k < states; k++)
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

                    for (k = 0; k < states; k++) {
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

                for (k = 0; k < states; k++) {
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

    for (i = 0; i < totalNodes; i++)
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
static void _updateInternalPllOnRatchet(pllInstance *tr, partitionList *pr) {
    //	cout << "lower = " << pr->partitionData[0]->lower << ", upper = " <<
    // pr->partitionData[0]->upper << ", aln->size() = " << iqtree->aln->size()
    // << endl;
    for (int i = 0; i < pr->numberOfPartitions; i++) {
        for (int ptn = pr->partitionData[i]->lower;
             ptn < pr->partitionData[i]->upper; ptn++) {
            tr->aliaswgt[ptn] = iqtree->aln->at(ptn).frequency;
        }
    }
}
static void compressDNA(pllInstance *tr, partitionList *pr, int *informative,
                        int perSiteScores) {
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

    for (model = 0; model < (size_t)pr->numberOfPartitions; model++) {
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

        for (i = lower; i < upper; i++)
            if (informative[i]) {
                entries += (size_t)tr->aliaswgt[i];
                pr->partitionData[model]->numInformativePatterns++;
            }

        compressedEntries = entries / PLL_PCF;

        if (entries % PLL_PCF != 0)
            compressedEntries++;

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

        rax_posix_memalign((void **)&(pr->partitionData[model]->parsVect),
                           PLL_BYTE_ALIGNMENT,
                           (size_t)compressedEntriesPadded * states *
                               totalNodes * sizeof(parsimonyNumber));

        for (i = 0; i < compressedEntriesPadded * states * totalNodes; i++)
            pr->partitionData[model]->parsVect[i] = 0;

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

        for (i = 0; i < (size_t)tr->mxtips; i++) {
            size_t w = 0, compressedIndex = 0, compressedCounter = 0, index = 0;

            for (k = 0; k < states; k++) {
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
                        for (k = 0; k < states; k++) {
                            if (value & mask32[k])
                                compressedValues[k] |=
                                    mask32[compressedCounter];
                        }

                        compressedCounter++;

                        if (compressedCounter == PLL_PCF) {
                            for (k = 0; k < states; k++) {
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
                    for (k = 0; k < states; k++)
                        compressedValues[k] |= mask32[compressedCounter];

                for (k = 0; k < states; k++) {
                    compressedTips[k][compressedIndex] = compressedValues[k];
                    compressedValues[k] = 0;
                }

                compressedCounter = 0;
            }
        }

        pr->partitionData[model]->parsimonyLength = compressedEntriesPadded;

        rax_free(compressedTips);
        rax_free(compressedValues);
    }

    rax_posix_memalign((void **)&(tr->parsimonyScore), PLL_BYTE_ALIGNMENT,
                       sizeof(unsigned int) * totalNodes);

    for (i = 0; i < totalNodes; i++)
        tr->parsimonyScore[i] = 0;
}

template <int perSiteScores>
void _allocateParsimonyDataStructuresTBR(pllInstance *tr, partitionList *pr) {
    int i;
    int *informative =
        (int *)rax_malloc(sizeof(int) * (size_t)tr->originalCrunchedLength);
    determineUninformativeSites(tr, pr, informative);

    if (pllCostMatrix) {
        for (int i = 0; i < pr->numberOfPartitions; i++) {
            pr->partitionData[i]->informativePtnWgt = NULL;
            pr->partitionData[i]->informativePtnScore = NULL;
        }
    }

    compressDNA(tr, pr, informative, perSiteScores);
    // cout << "Allocate parismony data structures\n";
    for (i = 1; i <= tr->mxtips + tr->mxtips - 2; i++) {
        nodeptr p = tr->nodep[i];
        p->recalculate = 0;
        p->xPars = 1;
        if (i > tr->mxtips) {
            p->next->xPars = 0;
            p->next->next->xPars = 0;
        }
    }

    tr->ti = (int *)rax_malloc(sizeof(int) * 4 * (size_t)tr->mxtips);

    rax_free(informative);
}

int pllSaveCurrentTreeTBRParsimony(pllInstance *tr, partitionList *pr,
                                   int cur_search_pars) {
    iqtree->saveCurrentTree(-cur_search_pars);
    return (int)(cur_search_pars);
}

/**
 * @brief Creates a bisection in the tree in the branch defined by the node p
 *
 * Splits the tree in two subtrees by removing the branch b(p<->p.back).
 *
 * @param tr, the tree
 * @param pr, the partitions
 * @param p, the node defining the branch to remove
 *
 * @return PLL_TRUE if OK, PLL_FALSE and sets errno in case of error
 */
static int pllTbrRemoveBranch(pllInstance *tr, partitionList *pr, nodeptr p,
                              bool insertNNI = false) {
    // int i;
    nodeptr p1, p2, q1, q2;
    // nodeptr tmpNode;

    // Evaluate pre-conditions
    // P1 : ( p in tr )

    // for (tmpNode = tr->start->next->back;
    //      tmpNode != tr->start && tmpNode != p;
    //      tmpNode = tmpNode->next->back) ;
    //
    // if(tmpNode != p) {
    //     // errno = PLL_TBR_INVALID_NODE;
    //     cout << "p is not in tr\n";
    //     return PLL_FALSE;
    // }

    // P2 : ( p is an inner branch )
    if (!(p->number > tr->mxtips && p->back->number > tr->mxtips)) {
        // errno = PLL_TBR_NOT_INNER_BRANCH;
        return PLL_FALSE;
    }

    p1 = p->next->back;
    p2 = p->next->next->back;
    q1 = p->back->next->back;
    q2 = p->back->next->next->back;

    if (insertNNI) {
        hookupDefault(p1, q1);
        hookupDefault(p2, q2);
    } else {
        hookupDefault(p1, p2);
        hookupDefault(q1, q2);
    }

    // Disconnect p->p* branch
    p->next->back = 0;
    p->next->next->back = 0;
    p->back->next->back = 0;
    p->back->next->next->back = 0;

    // Evaluate post-conditions?

    return PLL_TRUE;
}

static int pllTbrConnectSubtrees(pllInstance *tr, nodeptr p, nodeptr q,
                                 nodeptr *freeBranch, nodeptr *pb, nodeptr *qb,
                                 bool insertNNI = false) {
    int i;
    nodeptr tmpNode;

    *pb = 0;
    *qb = 0;

    // Evaluate preconditions

    // p and q must be connected and independent branches
    if (!(p && q && (p != q) && p->back && q->back && (p->back != q) &&
          (q->back != p))) {
        // errno = PLL_TBR_INVALID_NODE;
        return PLL_FALSE;
    }

    // p and q must belong to different subtrees. We check that we cannot
    // reach q starting from p

    // for (tmpNode = p->next->back; tmpNode != p &&
    // tmpNode != q;
    //     tmpNode = tmpNode->next->back)
    //   ;
    // if (tmpNode == q)
    //   {
    //     // p and q are in the same subtree
    //     // errno = PLL_TBR_INVALID_NODE;
    //     return PLL_FALSE;
    //   }

    (*pb) = p->back;
    (*qb) = q->back;
    tmpNode = 0;
    if (!freeBranch) {
        // Must exist an unconnected branch
        for (i = 1; i <= (2 * tr->mxtips - 3); i++) {
            if (!(tr->nodep[i]->back && tr->nodep[i]->next->back)) {
                tmpNode = tr->nodep[i];

                // It should have one and only one connected node
                if (tmpNode->next->back &&
                    !(tmpNode->back || tmpNode->next->next->back)) {
                    tmpNode = tmpNode->next->back;
                } else if (tmpNode->next->next->back &&
                           !(tmpNode->back || tmpNode->next->back)) {
                    tmpNode = tmpNode->next->next->back;
                } else if (!(tmpNode->back || tmpNode->next->back ||
                             tmpNode->next->next->back)) {
                    // There is no missing branch
                    // errno = PLL_TBR_INVALID_NODE;
                    return PLL_FALSE;
                }
                break;
            }
        }
    }

    if (!tmpNode && !freeBranch) {
        // There is no missing branch
        // errno = PLL_TBR_MISSING_FREE_BRANCH;
        return PLL_FALSE;
    }

    if (!freeBranch)
        (*freeBranch) = tmpNode;
    if ((*freeBranch)->back->xPars)
        (*freeBranch) = (*freeBranch)->back;
    assert((*freeBranch)->xPars);

    // Join subtrees
    if (insertNNI) {
        hookupDefault(p, (*freeBranch)->next);
        hookupDefault(q, (*freeBranch)->next->next);
        hookupDefault((*pb), (*freeBranch)->back->next);
        hookupDefault((*qb), (*freeBranch)->back->next->next);
    } else {
        hookupDefault(p, (*freeBranch)->next);
        hookupDefault((*pb), (*freeBranch)->next->next);
        hookupDefault(q, (*freeBranch)->back->next);
        hookupDefault((*qb), (*freeBranch)->back->next->next);
    }

    return PLL_TRUE;
}

static void reorderNodes(pllInstance *tr, nodeptr *np, nodeptr p, int *count,
                         bool resetParent = false) {
    if ((p->number <= tr->mxtips))
        return;
    else {
        tr->nodep[*count + tr->mxtips + 1] = p;
        *count = *count + 1;
        assert(p->xPars || resetParent);
        if (resetParent)
            p->next->back->par = p->next->next->back->par = p;

        reorderNodes(tr, np, p->next->back, count, resetParent);
        reorderNodes(tr, np, p->next->next->back, count, resetParent);
    }
}

static void nodeRectifierPars(pllInstance *tr, bool reset = false) {
    nodeptr *np = (nodeptr *)rax_malloc(2 * tr->mxtips * sizeof(nodeptr));
    int i;
    int count = 0;
    tr->start = tr->nodep[1];
    tr->rooted = PLL_FALSE;
    /* TODO why is tr->rooted set to PLL_FALSE here ?*/
    if (reset) {
        tr->curRoot = tr->nodep[1];
        tr->curRootBack = tr->nodep[1]->back;
        tr->curRoot->par = NULL;
        tr->start->back->par = tr->start;
    }

    for (i = tr->mxtips + 1; i <= (tr->mxtips + tr->mxtips - 1); i++)
        np[i] = tr->nodep[i];

    reorderNodes(tr, np, tr->curRoot, &count, reset);
    reorderNodes(tr, np, tr->curRoot->back, &count, reset);

    rax_free(np);
}

template <int perSiteScores>
static bool restoreTreeRearrangeParsimonyTBR(pllInstance *tr, partitionList *pr) {
    nodeptr q, r, rb, qb;

    if (!pllTbrRemoveBranch(tr, pr, tr->TBR_removeBranch, false)) {
        return PLL_FALSE;
    }
    q = tr->TBR_insertBranch1;
    r = tr->TBR_insertBranch2;
    q = (q->xPars ? q : q->back);
    r = (r->xPars ? r : r->back);
    if (!pllTbrConnectSubtrees(tr, q, r, &tr->TBR_removeBranch, &qb, &rb,
                               tr->TBR_insertNNI)) {
        return PLL_FALSE;
    }
    evaluateParsimonyTBR<perSiteScores>(tr, pr, q, r, tr->TBR_removeBranch);
    tr->curRoot = tr->TBR_removeBranch;
    tr->curRootBack = tr->TBR_removeBranch->back;

    return PLL_TRUE;
}

static void printTravInfo(int distInsBran1, int distInsBran2) {
    if (numDrawTrees == 10) {
        return;
    }
    ofstream out("TBRtree.txt", ios_base::app);
    out << "Distance between insertBranch1 and removeBranch (-1 means leaf): "
        << distInsBran1 << '\n';
    out << "Distance between insertBranch2 and removeBranch: " << distInsBran2
        << "\n\n";
    out.close();
}
static void updateLastTreeString(pllInstance *tr, partitionList *pr) {
    if (numDrawTrees == 10) {
        return;
    }
    pllTreeToNewick(tr->tree_string, tr, pr, tr->start->back, PLL_TRUE,
                    PLL_TRUE, 0, 0, 0, PLL_SUMMARIZE_LH, 0, 0);
    lastTreeString = string(tr->tree_string);
}
static void drawTreeTBR(pllInstance *tr, partitionList *pr) {
    if (numDrawTrees == 10)
        return;
    ofstream out;
    if (numDrawTrees == 0) {
        out.open("TBRtree.txt");
        out.close();
    }
    out.open("TBRtree.txt", ios_base::app);
    numDrawTrees++;
    double epsilon = 1.0 / iqtree->getAlnNSite();
    iqtree->readTreeString(lastTreeString);
    iqtree->initializeAllPartialPars();
    iqtree->clearAllPartialLH();
    int curScore = iqtree->computeParsimony();
    out << "TREE BEFORE ----------------------------------------------------: "
        << curScore << "\n";
    iqtree->sortTaxa();
    iqtree->drawTree(out, WT_BR_SCALE, epsilon);

    pllTreeToNewick(tr->tree_string, tr, pr, tr->start->back, PLL_TRUE,
                    PLL_TRUE, 0, 0, 0, PLL_SUMMARIZE_LH, 0, 0);
    string treeString = string(tr->tree_string);
    iqtree->readTreeString(treeString);
    iqtree->initializeAllPartialPars();
    iqtree->clearAllPartialLH();
    curScore = iqtree->computeParsimony();
    out << "TREE AFTER ----------------------------------------------------: "
        << curScore << "\n";

    iqtree->sortTaxa();
    iqtree->drawTree(out, WT_BR_SCALE, epsilon);
    out.close();
}

/** Based on PLL
 @brief Internal function for testing and saving a TBR move (if yeild better
 score)

 Checks the parsimony score when apply the given TBR move: Connect branch1 and
 branch2 together using freeBranch

 @param tr
 PLL instance

 @param pr
 List of partitions

 @param branch1
 Branch on one detached subtree

 @param branch2
 Branch on the other detached subtree

 @param freeBranch
 Branch that is disconnected before

 @param perSiteScores
 Calculate score for each site (Bootstrapping)

 @return
 PLL_TRUE if success, PLL_FALSE otherwise
 */
template <int perSiteScores>
static int pllTestTBRMove(pllInstance *tr, partitionList *pr, nodeptr branch1,
                          nodeptr branch2, nodeptr *freeBranch,
                          bool insertNNI = false) {
    int i;

    branch1 = (branch1->xPars ? branch1 : branch1->back);
    branch2 = (branch2->xPars ? branch2 : branch2->back);
    freeBranch = ((*freeBranch)->xPars ? freeBranch : (&((*freeBranch)->back)));
    // assert((*freeBranch)->xPars);
    nodeptr tmpNode = (insertNNI ? branch2 : branch1->back);
    nodeptr pb, qb;

    if (!pllTbrConnectSubtrees(tr, branch1, branch2, freeBranch, &pb, &qb,
                               insertNNI)) {
        cout << "Can't connect subtrees in test\n";
        return PLL_FALSE;
    }

    nodeptr TBR_removeBranch = NULL;
    if (branch1->back->next->back == tmpNode) {
        TBR_removeBranch = branch1->back->next->next;
    } else {
        TBR_removeBranch = branch1->back->next;
    }
    unsigned int mp = evaluateParsimonyTBR<perSiteScores>(tr, pr, branch1, branch2,
                                           TBR_removeBranch);
    tr->curRoot = TBR_removeBranch;
    tr->curRootBack = TBR_removeBranch->back;

    if (perSiteScores) {
        // If UFBoot is enabled ...
        pllSaveCurrentTreeTBRParsimony(tr, pr, mp); // run UFBoot
    }

    if (globalParam->tbr_test_draw == true) {
        drawTreeTBR(tr, pr);
    }
    if (mp < tr->bestParsimony)
        bestTreeScoreHits = 1;
    else if (mp == tr->bestParsimony)
        bestTreeScoreHits++;
    if ((mp < tr->bestParsimony) ||
        ((mp == tr->bestParsimony) &&
         (random_double() <= 1.0 / bestTreeScoreHits))) {
        tr->bestParsimony = mp;
        tr->TBR_insertBranch1 = branch1;
        tr->TBR_insertBranch2 = branch2;
        tr->TBR_removeBranch = TBR_removeBranch;
        tr->TBR_insertNNI = insertNNI;
    }

    /* restore */
    int restoreTopologyOK =
        pllTbrRemoveBranch(tr, pr, TBR_removeBranch, insertNNI);

    assert(restoreTopologyOK);

    return PLL_TRUE;
}

/**
 @brief Internal function for recursively traversing a tree and testing a
 possible TBR move insertion

 Recursively traverses the tree in direction of q (q->next->back and
 q->next->next->back) and at each (p, q) tests a TBR move between branches 'p'
 and 'q'.

 @note
 Version 1 is each P and Q has its own [mintrav, maxtrav]
 */
template <int perSiteScores>
static void pllTraverseUpdateTBRVer1Q(pllInstance *tr, partitionList *pr,
                                      nodeptr p, nodeptr q, nodeptr *r,
                                      int mintravQ, int maxtravQ) {

    if (mintravQ <= 0) {
        assert((pllTestTBRMove<perSiteScores>(tr, pr, p, q, r, false)));
        if (globalParam->tbr_insert_nni == true) {
            assert((pllTestTBRMove<perSiteScores>(tr, pr, p, q, r, true)));
        }
    }

    /* traverse the q subtree */
    if ((!isTip(q->number, tr->mxtips)) && (maxtravQ - 1 >= 0)) {
        pllTraverseUpdateTBRVer1Q<perSiteScores>(tr, pr, p, q->next->back, r, mintravQ - 1,
                                  maxtravQ - 1);
        pllTraverseUpdateTBRVer1Q<perSiteScores>(tr, pr, p, q->next->next->back, r,
                                  mintravQ - 1, maxtravQ - 1);
    }
}

/**
 @brief Internal function for recursively traversing a tree and testing a
 possible TBR move insertion

 Recursively traverses the tree in direction of p (p->next->back and
 p->next->next->back) and at each (p, q) tests a TBR move between branches 'p'
 and 'q'.

 @note
 Version 1 is each P and Q has its own [mintrav, maxtrav]
 */
template <int perSiteScores>
static void pllTraverseUpdateTBRVer1P(pllInstance *tr, partitionList *pr,
                                      nodeptr p, nodeptr q, nodeptr *r,
                                      int mintravP, int maxtravP, int mintravQ,
                                      int maxtravQ) {
    if (mintravP <= 0) {
        // Avoid insert back to where it's cut
        if (mintravP == 0 && mintravQ == 0) {
            if ((!isTip(q->number, tr->mxtips)) && (maxtravQ - 1 >= 0)) {
                pllTraverseUpdateTBRVer1Q<perSiteScores>(tr, pr, p, q->next->back, r,
                                          mintravQ - 1, maxtravQ - 1);
                pllTraverseUpdateTBRVer1Q<perSiteScores>(tr, pr, p, q->next->next->back, r,
                                          mintravQ - 1, maxtravQ - 1);
            }
        } else {
            pllTraverseUpdateTBRVer1Q<perSiteScores>(tr, pr, p, q, r, mintravQ, maxtravQ);
        }
    }
    /* traverse the p subtree */
    if (!isTip(p->number, tr->mxtips) && maxtravP - 1 >= 0) {
        pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p->next->back, q, r, mintravP - 1,
                                  maxtravP - 1, mintravQ, maxtravQ);
        pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p->next->next->back, q, r,
                                  mintravP - 1, maxtravP - 1, mintravQ,
                                  maxtravQ);
    }
}
/**
 @brief Internal function for recursively traversing a tree and testing a
 possible TBR move insertion

 Recursively traverses the tree in direction of q (q->next->back and
 q->next->next->back) and at each (p, q) tests a TBR move between branches 'p'
 and 'q'.

 @note
 Version 2 is Sum of distance of 2 inserted branch is in [mintrav, maxtrav]
 */
template <int perSiteScores>
static void pllTraverseUpdateTBRVer2Q(pllInstance *tr, partitionList *pr,
                                      nodeptr p, nodeptr q, nodeptr *r,
                                      int mintrav, int maxtrav, int distP,
                                      int distQ) {

    if (mintrav <= 0) {
        assert((pllTestTBRMove<perSiteScores>(tr, pr, p, q, r, false)));
        if (globalParam->tbr_insert_nni == true) {
            assert((pllTestTBRMove<perSiteScores>(tr, pr, p, q, r, true)));
        }
        if (globalParam->tbr_test_draw == true) {
            printTravInfo(distP, distQ);
        }
    }

    /* traverse the q subtree */
    if ((!isTip(q->number, tr->mxtips)) && (maxtrav - 1 >= 0)) {
        pllTraverseUpdateTBRVer2Q<perSiteScores>(tr, pr, p, q->next->back, r, mintrav - 1,
                                  maxtrav - 1, distP, distQ + 1);
        pllTraverseUpdateTBRVer2Q<perSiteScores>(tr, pr, p, q->next->next->back, r,
                                  mintrav - 1, maxtrav - 1, distP, distQ + 1);
    }
}

/**
 @brief Internal function for recursively traversing a tree and testing a
 possible TBR move insertion

 Recursively traverses the tree in direction of p (p->next->back and
 p->next->next->back) and at each (p, q) tests a TBR move between branches 'p'
 and 'q'.

 @note
 Version 2 is Sum of distance of 2 inserted branch is in [mintrav, maxtrav]
 */
template <int perSiteScores>
static void pllTraverseUpdateTBRVer2P(pllInstance *tr, partitionList *pr,
                                      nodeptr p, nodeptr q, nodeptr *r,
                                      int mintrav, int maxtrav, int distP,
                                      int distQ) {
    pllTraverseUpdateTBRVer2Q<perSiteScores>(tr, pr, p, q, r, mintrav, maxtrav, distP, distQ);
    /* traverse the p subtree */
    if ((!isTip(p->number, tr->mxtips)) && (maxtrav - 1 >= 0)) {
        pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p->next->back, q, r, mintrav - 1,
                                  maxtrav - 1, distP + 1, distQ);
        pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p->next->next->back, q, r,
                                  mintrav - 1, maxtrav - 1, distP + 1, distQ);
    }
}

/** Based on PLL
 @brief Find best TBR move given removeBranch

 Recursively tries all possible TBR moves that can be performed by
 pruning the branch at \a p and testing all possible 2 inserted branches
 in a distance of at least \a mintrav nodes and at most \a maxtrav nodes from
 each other

 @param tr
 PLL instance

 @param pr
 List of partitions

 @param p
 Node specifying the pruned branch.

 @param mintrav
 Minimum distance of 2 inserted branches

 @param maxtrav
 Maximum distance of 2 inserted branches

 @param perSiteScores
 Calculate scores for each site (Bootstrapping)

 @note
 Version 1 called pllTraverseUpdateTBRVer1 (default use Version 2)
 */
template <int perSiteScores>
static int pllComputeTBRVer1(pllInstance *tr, partitionList *pr, nodeptr p,
                             int mintrav, int maxtrav) {

    nodeptr p1, p2, q, q1, q2;
    int i, numPartitions;

    q = p->back;

    if (isTip(p->number, tr->mxtips) || isTip(q->number, tr->mxtips)) {
        // errno = PLL_TBR_NOT_INNER_BRANCH;
        return PLL_FALSE;
    }

    p1 = p->next->back;
    p2 = p->next->next->back;
    q1 = q->next->back;
    q2 = q->next->next->back;

    if (maxtrav < 1 || mintrav > maxtrav)
        return PLL_BADREAR;
    if (globalParam->tbr_test_draw == true) {
        updateLastTreeString(tr, pr);
    }
    /* split the tree in two components */
    if (!pllTbrRemoveBranch(tr, pr, p))
        return PLL_BADREAR;
    /* p1 and p2 are now connected */
    assert(p1->back == p2 && p2->back == p1);

    /* recursively traverse and perform TBR */
    pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p1, q1, &p, mintrav, maxtrav, mintrav,
                              maxtrav);
    if (!isTip(q2->number, tr->mxtips)) {
        pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p1, q2->next->back, &p, mintrav,
                                  maxtrav, mintrav - 1, maxtrav - 1);
        pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p1, q2->next->next->back, &p, mintrav,
                                  maxtrav, mintrav - 1, maxtrav - 1);
    }

    if (!isTip(p2->number, tr->mxtips)) {
        pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p2->next->back, q1, &p, mintrav - 1,
                                  maxtrav - 1, mintrav, maxtrav);
        pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p2->next->next->back, q1, &p,
                                  mintrav - 1, maxtrav - 1, mintrav, maxtrav);
        if (!isTip(q2->number, tr->mxtips)) {
            pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p2->next->back, q2->next->back,
                                      &p, mintrav - 1, maxtrav - 1, mintrav - 1,
                                      maxtrav - 1);
            pllTraverseUpdateTBRVer1P<perSiteScores>(
                tr, pr, p2->next->back, q2->next->next->back, &p, mintrav - 1,
                maxtrav - 1, mintrav - 1, maxtrav - 1);
            pllTraverseUpdateTBRVer1P<perSiteScores>(
                tr, pr, p2->next->next->back, q2->next->back, &p, mintrav - 1,
                maxtrav - 1, mintrav - 1, maxtrav - 1);
            pllTraverseUpdateTBRVer1P<perSiteScores>(tr, pr, p2->next->next->back,
                                      q2->next->next->back, &p, mintrav - 1,
                                      maxtrav - 1, mintrav - 1, maxtrav - 1);
        }
    }
    nodeptr pb, qb, freeBranch;
    /* restore the topology as it was before the split */
    freeBranch = (p->xPars ? p : q);
    p1 = (p1->xPars ? p1 : p1->back);
    q1 = (q1->xPars ? q1 : q1->back);
    int restoreTopoOK =
        pllTbrConnectSubtrees(tr, p1, q1, &freeBranch, &pb, &qb);
    evaluateParsimonyTBR<perSiteScores>(tr, pr, p1, q1, freeBranch);
    tr->curRoot = freeBranch;
    tr->curRootBack = freeBranch->back;
    assert(restoreTopoOK);

    return PLL_TRUE;
}

/** Based on PLL
 @brief Find best TBR move given removeBranch

 Recursively tries all possible TBR moves that can be performed by
 pruning the branch at \a p and testing all possible 2 inserted branches
 in a distance of at least \a mintrav nodes and at most \a maxtrav nodes from
 each other

 @param tr
 PLL instance

 @param pr
 List of partitions

 @param p
 Node specifying the pruned branch.

 @param mintrav
 Minimum distance of 2 inserted branches

 @param maxtrav
 Maximum distance of 2 inserted branches

 @param perSiteScores
 Calculate scores for each site (Bootstrapping)

 @note
 Version 2 called pllTraverseUpdateTBRVer2 (default use version 2).
 */
template <int perSiteScores>
static int pllComputeTBRVer2(pllInstance *tr, partitionList *pr, nodeptr p,
                             int mintrav, int maxtrav) {

    nodeptr p1, p2, q, q1, q2;
    int i, numPartitions;

    q = p->back;

    if (isTip(p->number, tr->mxtips) || isTip(q->number, tr->mxtips)) {
        // errno = PLL_TBR_NOT_INNER_BRANCH;
        return PLL_FALSE;
    }

    p1 = p->next->back;
    p2 = p->next->next->back;
    q1 = q->next->back;
    q2 = q->next->next->back;

    if (maxtrav < 1 || mintrav > maxtrav)
        return PLL_BADREAR;
    if (globalParam->tbr_test_draw == true) {
        updateLastTreeString(tr, pr);
    }
    /* split the tree in two components */
    if (!pllTbrRemoveBranch(tr, pr, p))
        return PLL_BADREAR;
    /* p1 and p2 are now connected */
    assert(p1->back == p2 && p2->back == p1);

    /* recursively traverse and perform TBR */
    pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p1, q1, &p, mintrav, maxtrav, 0, 0);
    if (!isTip(q2->number, tr->mxtips)) {
        pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p1, q2->next->back, &p, mintrav - 1,
                                  maxtrav - 1, 0, 1);
        pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p1, q2->next->next->back, &p,
                                  mintrav - 1, maxtrav - 1, 0, 1);
    }

    if (!isTip(p2->number, tr->mxtips)) {
        pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p2->next->back, q1, &p, mintrav - 1,
                                  maxtrav - 1, 1, 0);
        pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p2->next->next->back, q1, &p,
                                  mintrav - 1, maxtrav - 1, 1, 0);
        if (!isTip(q2->number, tr->mxtips)) {
            pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p2->next->back, q2->next->back,
                                      &p, mintrav - 2, maxtrav - 2, 1, 1);
            pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p2->next->back,
                                      q2->next->next->back, &p, mintrav - 2,
                                      maxtrav - 2, 1, 1);
            pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p2->next->next->back,
                                      q2->next->back, &p, mintrav - 2,
                                      maxtrav - 2, 1, 1);
            pllTraverseUpdateTBRVer2P<perSiteScores>(tr, pr, p2->next->next->back,
                                      q2->next->next->back, &p, mintrav - 2,
                                      maxtrav - 2, 1, 1);
        }
    }
    nodeptr pb, qb, freeBranch;
    /* restore the topology as it was before the split */
    freeBranch = (p->xPars ? p : q);
    p1 = (p1->xPars ? p1 : p1->back);
    q1 = (q1->xPars ? q1 : q1->back);
    int restoreTopoOK =
        pllTbrConnectSubtrees(tr, p1, q1, &freeBranch, &pb, &qb);
    evaluateParsimonyTBR<perSiteScores>(tr, pr, p1, q1, freeBranch);
    tr->curRoot = freeBranch;
    tr->curRootBack = freeBranch->back;
    assert(restoreTopoOK);

    return PLL_TRUE;
}

template <int perSiteScores>
static int pllTestTBRMoveLeaf(pllInstance *tr, partitionList *pr,
                              nodeptr insertBranch, nodeptr removeBranch) {
    insertBranch = (insertBranch->xPars ? insertBranch : insertBranch->back);
    removeBranch = (removeBranch->xPars ? removeBranch : removeBranch->back);
    nodeptr p = (isTip(removeBranch->number, tr->mxtips) ? removeBranch->back
                                                         : removeBranch);
    // Connect
    nodeptr i1 = insertBranch;
    nodeptr i2 = i1->back;
    hookupDefault(i1, p->next);
    hookupDefault(i2, p->next->next);
    unsigned int mp =
        evaluateParsimonyTBR<perSiteScores>(tr, pr, p->back, insertBranch, p);
    tr->curRoot = removeBranch;
    tr->curRootBack = removeBranch->back;
    if (globalParam->tbr_test_draw == true) {
        drawTreeTBR(tr, pr);
    }
    if (mp < tr->bestParsimony)
        bestTreeScoreHits = 1;
    else if (mp == tr->bestParsimony)
        bestTreeScoreHits++;

    if ((mp < tr->bestParsimony) ||
        ((mp == tr->bestParsimony) &&
         (random_double() <= 1.0 / bestTreeScoreHits))) {
        tr->bestParsimony = mp;
        tr->TBR_insertBranch1 =
            (insertBranch->xPars ? insertBranch : insertBranch->back);
        tr->TBR_removeBranch = p;
    }

    // Remove
    hookupDefault(p->next->back, p->next->next->back);
    p->next->back = p->next->next->back = NULL;

    return PLL_TRUE;
}

template <int perSiteScores>
static void pllTraverseUpdateTBRLeaf(pllInstance *tr, partitionList *pr,
                                     nodeptr p, nodeptr removeBranch,
                                     int mintrav, int maxtrav, int distP) {
    if (mintrav <= 0) {
        assert(pllTestTBRMoveLeaf<perSiteScores>(tr, pr, p, removeBranch));
        if (globalParam->tbr_test_draw == true) {
            printTravInfo(-1, distP);
        }
    }
    if (!isTip(p->number, tr->mxtips) && maxtrav - 1 >= 0) {
        pllTraverseUpdateTBRLeaf<perSiteScores>(tr, pr, p->next->back, removeBranch,
                                 mintrav - 1, maxtrav - 1, distP + 1);
        pllTraverseUpdateTBRLeaf<perSiteScores>(tr, pr, p->next->next->back, removeBranch,
                                 mintrav - 1, maxtrav - 1, distP + 1);
    }
}

template <int perSiteScores>
static int pllComputeTBRLeaf(pllInstance *tr, partitionList *pr, nodeptr p,
                             int mintrav, int maxtrav) {
    nodeptr q = p->back;
    if (!isTip(q->number, tr->mxtips)) {
        swap(p, q);
        // q must be leaf
    }

    if (!isTip(q->number, tr->mxtips)) {
        // Both p and q are not leaves.
        return PLL_FALSE;
    }
    nodeptr p1, p2;
    p1 = p->next->back;
    p2 = p->next->next->back;

    if (globalParam->tbr_test_draw == true) {
        updateLastTreeString(tr, pr);
    }
    // Disconnect edge (p, p1) and (p, p2)
    // Connect (p1, p2)
    hookupDefault(p1, p2);
    p->next->back = p->next->next->back = NULL;

    if (!isTip(p1->number, tr->mxtips)) {
        pllTraverseUpdateTBRLeaf<perSiteScores>(tr, pr, p1->next->back, p, mintrav - 1,
                                 maxtrav - 1, 1);
        pllTraverseUpdateTBRLeaf<perSiteScores>(tr, pr, p1->next->next->back, p, mintrav - 1,
                                 maxtrav - 1, 1);
    }

    if (!isTip(p2->number, tr->mxtips)) {
        pllTraverseUpdateTBRLeaf<perSiteScores>(tr, pr, p2->next->back, p, mintrav - 1,
                                 maxtrav - 1, 1);
        pllTraverseUpdateTBRLeaf<perSiteScores>(tr, pr, p2->next->next->back, p, mintrav - 1,
                                 maxtrav - 1, 1);
    }

    // Connect p to p1 and p2 again
    hookupDefault(p->next, p1);
    hookupDefault(p->next->next, p2);
    p1 = (p1->xPars ? p1 : p2);
    // assert(p1->xPars);
    evaluateParsimonyTBR<perSiteScores>(tr, pr, q, p1, q);
    tr->curRoot = q;
    tr->curRootBack = q->back;
    return PLL_TRUE;
}

template <int perSiteScores>
static bool restoreTreeRearrangeParsimonyTBRLeaf(pllInstance *tr,
                                                 partitionList *pr) {

    hookupDefault(tr->TBR_removeBranch->next->back,
                  tr->TBR_removeBranch->next->next->back);

    nodeptr r = tr->TBR_insertBranch1;
    nodeptr rb = r->back;
    if (!r->xPars) {
        swap(r, rb);
    }
    assert(r->xPars);
    assert(tr->TBR_removeBranch->xPars);
    hookupDefault(r, tr->TBR_removeBranch->next);
    hookupDefault(rb, tr->TBR_removeBranch->next->next);
    evaluateParsimonyTBR<perSiteScores>(tr, pr, tr->TBR_removeBranch->back, r,
                         tr->TBR_removeBranch);
    tr->curRoot = tr->TBR_removeBranch;
    tr->curRootBack = tr->TBR_removeBranch->back;

    return PLL_TRUE;
}

void testTBROnUserTree(Params &params) {}
/*
{
        Alignment alignment(params.aln_file, params.sequence_type,
params.intype);

  IQTree * ptree = new IQTree(&alignment);
  (ptree->params) = &params;
  ofstream out("tbr_test.txt");

        cout << "Read user tree... 1st time";
  ptree->readTree(params.user_file, params.is_rooted);

        ptree->setAlignment(&alignment); // IMPORTANT: Always call
setAlignment() after readTree() optimizeAlignment(ptree, params);

        cout << "Read user tree... 2nd time\n";
  // ptree->readTree(params.user_file, params.is_rooted);

        // ptree->setAlignment(ptree->aln); // IMPORTANT: Always call
setAlignment() after readTree() ptree->initializeAllPartialPars();
  ptree->clearAllPartialLH();
  ptree->initializePLL(params);
  string tree_string = ptree->getTreeString();
  pllNewickTree *pll_tree = pllNewickParseString(tree_string.c_str());
  assert(pll_tree != NULL);
  pllTreeInitTopologyNewick(ptree->pllInst, pll_tree, PLL_FALSE);
  pllNewickParseDestroy(&pll_tree);
  _allocateParsimonyDataStructuresTBR(ptree->pllInst, ptree->pllPartitions,
false); nodeRectifierPars(ptree->pllInst); ptree->pllInst->bestParsimony =
UINT_MAX; // Important because of early termination in
evaluateSankoffParsimonyIterativeFastSIMD ptree->pllInst->bestParsimony =
_evaluateParsimony(ptree->pllInst, ptree->pllPartitions, ptree->pllInst->start,
PLL_TRUE, false); double epsilon = 1.0 / ptree->getAlnNSite(); out << "Tree
before 1 TBR looks like: \n"; ptree->sortTaxa(); ptree->drawTree(out,
WT_BR_SCALE, epsilon); out << "Parsimony score: " <<
ptree->pllInst->bestParsimony << endl; unsigned int curScore =
ptree->pllInst->bestParsimony; ptree->pllInst->ntips = ptree->pllInst->mxtips;
  ptree->pllInst->TBR_removeBranch = NULL;
  ptree->pllInst->TBR_insertBranch1 = NULL;
  ptree->pllInst->TBR_insertBranch2 = NULL;

  // TBR

  int ok = pllComputeTBR (ptree->pllInst, ptree->pllPartitions,
ptree->pllInst->nodep[23], params.spr_mintrav, params.spr_maxtrav); cout << "OK:
" << ok << '\n'; cout << ptree->pllInst->bestParsimony << ' ' << curScore <<
'\n'; if (ptree->pllInst->bestParsimony != curScore)
  {
    cout << "Found better\n";
    assert(restoreTreeRearrangeParsimonyTBR(ptree->pllInst,
ptree->pllPartitions, 0));
  }
  pllTreeToNewick(ptree->pllInst->tree_string, ptree->pllInst,
ptree->pllPartitions, ptree->pllInst->start->back, PLL_TRUE, PLL_TRUE, 0, 0, 0,
PLL_SUMMARIZE_LH, 0, 0); string treeString =
string(ptree->pllInst->tree_string); ptree->readTreeString(treeString);
  ptree->initializeAllPartialPars();
  ptree->clearAllPartialLH();
  curScore = ptree->computeParsimony();
  // assert(curScore == ptree->pllInst->bestParsimony);
  out << "Tree after 1 TBR looks like: \n";
  ptree->sortTaxa();
  ptree->drawTree(out, WT_BR_SCALE, epsilon);
  out << "Parsimony score: " << curScore << endl;
  _pllFreeParsimonyDataStructures(ptree->pllInst, ptree->pllPartitions);
  cout << "Finished\n";
  delete ptree;
}
*/

int pllOptimizeTbrParsimony(pllInstance *tr, partitionList *pr, int mintrav,
                            int maxtrav, IQTree *_iqtree) {
    int perSiteScores = globalParam->gbo_replicates > 0;

    iqtree = _iqtree; // update pointer to IQTree

    if (globalParam->ratchet_iter >= 0 &&
        (iqtree->on_ratchet_hclimb1 || iqtree->on_ratchet_hclimb2)) {
        // oct 23: in non-ratchet iteration, allocate is not triggered
        _updateInternalPllOnRatchet(tr, pr);
        if (perSiteScores) {
            _allocateParsimonyDataStructuresTBR<true>(tr, pr);
        } else {
            _allocateParsimonyDataStructuresTBR<false>(
                tr, pr); // called once if not running ratchet
        }
    } else if (first_call || (iqtree && iqtree->on_opt_btree)) {
        if (perSiteScores) {
            _allocateParsimonyDataStructuresTBR<true>(tr, pr);
        } else {
            _allocateParsimonyDataStructuresTBR<false>(
                tr, pr); // called once if not running ratchet   
        }
    }
    if (first_call) {
        first_call = false;
    }

    int i;
    unsigned int randomMP, startMP;

    assert(!tr->constrained);

    nodeRectifierPars(tr, true);
    tr->bestParsimony = UINT_MAX;
    if (perSiteScores) {
        tr->bestParsimony =
            _evaluateParsimony<true>(tr, pr, tr->start, PLL_TRUE);
    } else {
        tr->bestParsimony =
            _evaluateParsimony<false>(tr, pr, tr->start, PLL_TRUE);
    }

    assert(-iqtree->curScore == tr->bestParsimony);

    unsigned int bestIterationScoreHits = 1;
    randomMP = tr->bestParsimony;

    do {
        nodeRectifierPars(tr, false);
        startMP = randomMP;

        for (i = 1; i <= tr->mxtips; i++) {
            tr->TBR_removeBranch = tr->TBR_insertBranch1 = NULL;
            tr->TBR_insertNNI = false;
            bestTreeScoreHits = 1;
            if (perSiteScores) {
                pllComputeTBRLeaf<true>(tr, pr, tr->nodep[i]->back, mintrav, maxtrav);
            } else {
                pllComputeTBRLeaf<false>(tr, pr, tr->nodep[i]->back, mintrav, maxtrav);
            }
            if (tr->bestParsimony == randomMP)
                bestIterationScoreHits++;
            if (tr->bestParsimony < randomMP)
                bestIterationScoreHits = 1;
            if (((tr->bestParsimony < randomMP) ||
                 ((tr->bestParsimony == randomMP) &&
                  (random_double() <= 1.0 / bestIterationScoreHits))) &&
                tr->TBR_removeBranch && tr->TBR_insertBranch1) {
                if (perSiteScores) {
                    restoreTreeRearrangeParsimonyTBRLeaf<true>(tr, pr);
                } else {
                    restoreTreeRearrangeParsimonyTBRLeaf<false>(tr, pr);
                }
                randomMP = tr->bestParsimony;
            }
        }

        for (i = tr->mxtips + 1; i <= tr->mxtips + tr->mxtips - 2; i++) {
            tr->TBR_removeBranch = NULL;
            tr->TBR_insertBranch1 = tr->TBR_insertBranch2 = NULL;
            tr->TBR_insertNNI = false;
            bestTreeScoreHits = 1;
            if (globalParam->tbr_traverse_ver1 == true) {
                if (perSiteScores) {
                    pllComputeTBRVer1<true>(tr, pr, tr->nodep[i], mintrav, maxtrav);
                } else {
                    pllComputeTBRVer1<false>(tr, pr, tr->nodep[i], mintrav, maxtrav);
                }
            } else {
                if (perSiteScores) {
                    pllComputeTBRVer2<true>(tr, pr, tr->nodep[i], mintrav, maxtrav);
                } else {
                    pllComputeTBRVer2<false>(tr, pr, tr->nodep[i], mintrav, maxtrav);
                }
            }
            if (tr->bestParsimony == randomMP)
                bestIterationScoreHits++;
            if (tr->bestParsimony < randomMP)
                bestIterationScoreHits = 1;
            if (((tr->bestParsimony < randomMP) ||
                 ((tr->bestParsimony == randomMP) &&
                  (random_double() <= 1.0 / bestIterationScoreHits))) &&
                tr->TBR_removeBranch && tr->TBR_insertBranch1 &&
                tr->TBR_insertBranch2) {
                if (perSiteScores) {
                    restoreTreeRearrangeParsimonyTBR<true>(tr, pr);
                } else {
                    restoreTreeRearrangeParsimonyTBR<false>(tr, pr);
                }
                randomMP = tr->bestParsimony;
            }
        }
    } while (randomMP < startMP);
    return startMP;
}

static void makePermutationFast(int *perm, int n, pllInstance *tr) {
    int i, j, k;

    for (i = 1; i <= n; i++)
        perm[i] = i;

    for (i = 1; i <= n; i++) {
        double d = randum(&tr->randomNumberSeed);

        k = (int)((double)(n + 1 - i) * d);

        j = perm[i];

        perm[i] = perm[i + k];
        perm[i + k] = j;
    }
}

template <int perSiteScores>
static void insertParsimony(pllInstance *tr, partitionList *pr, nodeptr p,
                            nodeptr q) {
    nodeptr r;

    r = q->back;

    hookupDefault(p->next, q);
    hookupDefault(p->next->next, r);
    _newviewParsimony<perSiteScores>(tr, pr, p);
}

static nodeptr buildNewTip(pllInstance *tr, nodeptr p) {
    nodeptr q;

    q = tr->nodep[(tr->nextnode)++];
    hookupDefault(p, q);
    q->next->back = (nodeptr)NULL;
    q->next->next->back = (nodeptr)NULL;

    return q;
}

static void buildSimpleTree(pllInstance *tr, partitionList *pr, int ip, int iq,
                            int ir) {
    nodeptr p, s;
    int i;

    i = PLL_MIN(ip, iq);
    if (ir < i)
        i = ir;
    tr->start = tr->nodep[i];
    tr->ntips = 3;
    p = tr->nodep[ip];
    hookupDefault(p, tr->nodep[iq]);
    s = buildNewTip(tr, tr->nodep[ir]);
    insertParsimony<PLL_FALSE>(tr, pr, s, p);
}

static void stepwiseAddition(pllInstance *tr, partitionList *pr, nodeptr p,
                             nodeptr q) {
    nodeptr r = q->back;

    unsigned int mp;

    int counter = 4;

    p->next->back = q;
    q->back = p->next;

    p->next->next->back = r;
    r->back = p->next->next;

    computeTraversalInfoParsimony<PLL_FALSE>(p, tr->ti, &counter, tr->mxtips, PLL_FALSE);
    tr->ti[0] = counter;
    tr->ti[1] = p->number;
    tr->ti[2] = p->back->number;

    mp = _evaluateParsimonyIterativeFast<PLL_FALSE>(tr, pr);

    if (mp < tr->bestParsimony)
        bestTreeScoreHits = 1;
    else if (mp == tr->bestParsimony)
        bestTreeScoreHits++;

    if ((mp < tr->bestParsimony) ||
        ((mp == tr->bestParsimony) &&
         (random_double() <= 1.0 / bestTreeScoreHits))) {
        tr->bestParsimony = mp;
        tr->insertNode = q;
    }

    q->back = r;
    r->back = q;

    // TODO: why need parsimonyScore here?
    if (q->number > tr->mxtips && tr->parsimonyScore[q->number] > 0) {
        stepwiseAddition(tr, pr, p, q->next->back);
        stepwiseAddition(tr, pr, p, q->next->next->back);
    }
}

static void pllMakeParsimonyTreeFastTBR(pllInstance *tr, partitionList *pr,
                                        int tbr_mintrav, int tbr_maxtrav) {
    nodeptr p, f;
    int i, nextsp,
        *perm = (int *)rax_malloc((size_t)(tr->mxtips + 1) * sizeof(int));

    unsigned int randomMP, startMP;

    assert(!tr->constrained);

    makePermutationFast(perm, tr->mxtips, tr);

    tr->ntips = 0;

    tr->nextnode = tr->mxtips + 1;

    buildSimpleTree(tr, pr, perm[1], perm[2], perm[3]);

    f = tr->start;

    bestTreeScoreHits = 1;
    while (tr->ntips < tr->mxtips) {
        nodeptr q;

        tr->bestParsimony = INT_MAX;
        nextsp = ++(tr->ntips);
        p = tr->nodep[perm[nextsp]];
        q = tr->nodep[(tr->nextnode)++];
        p->back = q;
        q->back = p;

        stepwiseAddition(tr, pr, q, f->back);
        //      cout << "tr->ntips = " << tr->ntips << endl;

        {
            nodeptr r = tr->insertNode->back;

            int counter = 4;

            hookupDefault(q->next, tr->insertNode);
            hookupDefault(q->next->next, r);

            computeTraversalInfoParsimony<0>(q, tr->ti, &counter, tr->mxtips,
                                          PLL_FALSE);
            tr->ti[0] = counter;

            _newviewParsimonyIterativeFast<0>(tr, pr);
        }
    }
    rax_free(perm);
    nodeRectifierPars(tr, true);

    tr->bestParsimony = UINT_MAX;
    tr->bestParsimony =
        _evaluateParsimony<PLL_FALSE>(tr, pr, tr->start, PLL_TRUE);

    unsigned int bestIterationScoreHits = 1;
    randomMP = tr->bestParsimony;
    do {
        nodeRectifierPars(tr, false);
        startMP = randomMP;

        for (i = 1; i <= tr->mxtips; i++) {
            tr->TBR_removeBranch = tr->TBR_insertBranch1 = NULL;
            bestTreeScoreHits = 1;
            pllComputeTBRLeaf<PLL_FALSE>(tr, pr, tr->nodep[i]->back, tbr_mintrav,
                              tbr_maxtrav);
            if (tr->bestParsimony == randomMP)
                bestIterationScoreHits++;
            if (tr->bestParsimony < randomMP)
                bestIterationScoreHits = 1;
            if (((tr->bestParsimony < randomMP) ||
                 ((tr->bestParsimony == randomMP) &&
                  (random_double() <= 1.0 / bestIterationScoreHits))) &&
                tr->TBR_removeBranch && tr->TBR_insertBranch1) {
                restoreTreeRearrangeParsimonyTBRLeaf<PLL_FALSE>(tr, pr);
                randomMP = tr->bestParsimony;
            }
        }

        for (i = tr->mxtips + 1; i <= tr->mxtips + tr->mxtips - 2; i++) {
            //		for(j = 1; j <= tr->mxtips + tr->mxtips - 2;
            // j++){ 			i = perm[j];
            tr->TBR_removeBranch = NULL;
            tr->TBR_insertBranch1 = tr->TBR_insertBranch2 = NULL;
            bestTreeScoreHits = 1;
            // assert(tr->nodep[i]->xPars);
            if (globalParam->tbr_traverse_ver1 == true) {
                pllComputeTBRVer1<PLL_FALSE>(tr, pr, tr->nodep[i], tbr_mintrav,
                                  tbr_maxtrav);
            } else {
                pllComputeTBRVer2<PLL_FALSE>(tr, pr, tr->nodep[i], tbr_mintrav,
                                  tbr_maxtrav);
            }
            if (tr->bestParsimony == randomMP)
                bestIterationScoreHits++;
            if (tr->bestParsimony < randomMP)
                bestIterationScoreHits = 1;
            if (((tr->bestParsimony < randomMP) ||
                 ((tr->bestParsimony == randomMP) &&
                  (random_double() <= 1.0 / bestIterationScoreHits))) &&
                tr->TBR_removeBranch && tr->TBR_insertBranch1 &&
                tr->TBR_insertBranch2) {
                restoreTreeRearrangeParsimonyTBR<PLL_FALSE>(tr, pr);
                randomMP = tr->bestParsimony;
            }
        }
    } while (randomMP < startMP);
}

/** @brief Compute a randomized stepwise addition oder parsimony tree

    Implements the RAxML randomized stepwise addition order algorithm

    @todo
      check functions that are invoked for potential memory leaks!

    @param tr
      The PLL instance

    @param partitions
      The partitions
*/
void pllComputeRandomizedStepwiseAdditionParsimonyTreeTBR(
    pllInstance *tr, partitionList *partitions, int tbr_mintrav,
    int tbr_maxtrav, IQTree *_iqtree) {
    doing_stepwise_addition = true;
    iqtree = _iqtree; // update pointer to IQTree
    _allocateParsimonyDataStructuresTBR<PLL_FALSE>(tr, partitions);
    //	cout << "DONE allocate..." << endl;
    pllMakeParsimonyTreeFastTBR(tr, partitions, tbr_mintrav, tbr_maxtrav);
    //	cout << "DONE make...." << endl;
    _pllFreeParsimonyDataStructures(tr, partitions);
    doing_stepwise_addition = false;
    //	cout << "Done free..." << endl;
}
