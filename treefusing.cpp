//
// Created on 24/10/2024.
//
#include "treefusing.h"
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
#define CAST double*
#define SET_ALL_BITS_ONE _mm512_set1_epi32(0xFFFFFFFF)
#define SET_ALL_BITS_ZERO _mm512_setzero_epi32()
#define VECTOR_LOAD _mm512_load_epi32
#define VECTOR_STORE  _mm512_store_epi32
#define VECTOR_BIT_AND _mm512_and_epi32
#define VECTOR_BIT_OR  _mm512_or_epi32
#define VECTOR_AND_NOT _mm512_andnot_epi32

#elif defined(__AVX)

#include <xmmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "vectorclass/vectorclass.h"

#define VECTOR_SIZE 8
#define ULINT_SIZE 64
#define USHORT_PER_VECTOR 16
#define INTS_PER_VECTOR 8
#define LONG_INTS_PER_VECTOR 4
//#define LONG_INTS_PER_VECTOR (32/sizeof(long))
#define INT_TYPE __m256d
#define CAST double*
#define SET_ALL_BITS_ONE (__m256d)_mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
#define SET_ALL_BITS_ZERO (__m256d)_mm256_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000)
#define VECTOR_LOAD _mm256_load_pd
#define VECTOR_BIT_AND _mm256_and_pd
#define VECTOR_BIT_OR  _mm256_or_pd
#define VECTOR_STORE  _mm256_store_pd
#define VECTOR_AND_NOT _mm256_andnot_pd

#elif (defined(__SSE3))

#include <xmmintrin.h>
#include <pmmintrin.h>
#include "vectorclass/vectorclass.h"

#define VECTOR_SIZE 4
#define USHORT_PER_VECTOR 8
#define INTS_PER_VECTOR 4
#ifdef __i386__
#   define ULINT_SIZE 32
#   define LONG_INTS_PER_VECTOR 4
//#define LONG_INTS_PER_VECTOR (16/sizeof(long))
#else
#   define ULINT_SIZE 64
#   define LONG_INTS_PER_VECTOR 2
//#define LONG_INTS_PER_VECTOR (16/sizeof(long))
#endif
#define INT_TYPE __m128i
#define CAST __m128i*
#define SET_ALL_BITS_ONE _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
#define SET_ALL_BITS_ZERO _mm_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000)
#define VECTOR_LOAD _mm_load_si128
#define VECTOR_BIT_AND _mm_and_si128
#define VECTOR_BIT_OR  _mm_or_si128
#define VECTOR_STORE  _mm_store_si128
#define VECTOR_AND_NOT _mm_andnot_si128

#else
// no vectorization
#define VECTOR_SIZE 1
#endif

#include "pllrepo/src/pll.h"
#include "pllrepo/src/pllInternal.h"

static pllBoolean tipHomogeneityCheckerPars(pllInstance *tr, nodeptr p, int grouping);

extern const unsigned int mask32[32];
/* vector-specific stuff */


extern double masterTime;

/* program options */
extern Params *globalParam;
static IQTree * iqtree = NULL;
static unsigned long bestTreeScoreHits; // to count hits to bestParsimony

extern parsimonyNumber * pllCostMatrix; // Diep: For weighted version
extern int pllCostNstates; // Diep: For weighted version
extern parsimonyNumber *vectorCostMatrix; // BQM: vectorized cost matrix
static parsimonyNumber highest_cost;

//(if needed) split the parsimony vector into several segments to avoid overflow when calc rell based on vec8us
extern int pllRepsSegments; // # of segments
extern int * pllSegmentUpper; // array of first index of the next segment, see IQTree::segment_upper
static parsimonyNumber * pllRemainderLowerBounds; // array of lower bound score for the un-calculated part to the right of a segment
static bool first_call = true; // is this the first call to pllOptimizeSprParsimony
static bool doing_stepwise_addition = false; // is the stepwise addition on

static void resetGlobalParamOnNewAln(){
    globalParam = NULL;
    iqtree = NULL;
    bestTreeScoreHits = 0;
    pllCostMatrix = NULL;
    pllCostNstates = 0;
    vectorCostMatrix = NULL;
    highest_cost = 0;

    pllRepsSegments = -1;
    pllSegmentUpper = NULL;
    pllRemainderLowerBounds = NULL;
    first_call = true;
    doing_stepwise_addition = false;
}

static void initializeCostMatrix() {
    highest_cost = *max_element(pllCostMatrix, pllCostMatrix+pllCostNstates*pllCostNstates) + 1;

//    cout << "Segments: ";
//    for (int i = 0; i < pllRepsSegments; i++)
//        cout <<  " " << pllSegmentUpper[i];
//    cout << endl;

#if (defined(__SSE3) || defined(__AVX))
    assert(pllCostMatrix);
    if (!vectorCostMatrix) {
        rax_posix_memalign ((void **) &(vectorCostMatrix), PLL_BYTE_ALIGNMENT, sizeof(parsimonyNumber)*pllCostNstates*pllCostNstates);

        if (globalParam->sankoff_short_int) {
            parsimonyNumberShort *shortMatrix = (parsimonyNumberShort*)vectorCostMatrix;
            // duplicate the cost entries for vector operations
            for (int i = 0; i < pllCostNstates; i++)
                for (int j = 0; j < pllCostNstates; j++)
                    shortMatrix[(i*pllCostNstates+j)] = pllCostMatrix[i*pllCostNstates+j];
        } else {
            // duplicate the cost entries for vector operations
            for (int i = 0; i < pllCostNstates; i++)
                for (int j = 0; j < pllCostNstates; j++)
                    vectorCostMatrix[(i*pllCostNstates+j)] = pllCostMatrix[i*pllCostNstates+j];
        }
    }
#else
    vectorCostMatrix = NULL;
#endif
}

// note: pllCostMatrix[i*pllCostNstates+j] = cost from i to j

///************************************************ pop count stuff ***********************************************/
//
// unsigned int bitcount_32_bit(unsigned int i)
//{
//  return ((unsigned int) __builtin_popcount(i));
//}

///* bit count for 64 bit integers */
//
//inline unsigned int bitcount_64_bit(unsigned long i)
//{
//  return ((unsigned int) __builtin_popcountl(i));
//}

/* bit count for 128 bit SSE3 and 256 bit AVX registers */

#if (defined(__SSE3) || defined(__AVX))

#ifdef _WIN32
/* emulate with 32-bit version */
static __inline unsigned int vectorPopcount(INT_TYPE v)
{
PLL_ALIGN_BEGIN unsigned int counts[INTS_PER_VECTOR] PLL_ALIGN_END;

  int
    i,
    sum = 0;

  VECTOR_STORE((CAST)counts, v);

  for(i = 0; i < INTS_PER_VECTOR; i++)
    sum += __builtin_popcount(counts[i]);
  // cout<<sum<<"hihihi"<<endl;
  return ((unsigned int)sum);
}
#else

static inline unsigned int vectorPopcount(INT_TYPE v)
{
    unsigned long
            counts[LONG_INTS_PER_VECTOR] __attribute__ ((aligned (PLL_BYTE_ALIGNMENT)));

    int
            i,
            sum = 0;

    VECTOR_STORE((CAST)counts, v);

    for(i = 0; i < LONG_INTS_PER_VECTOR; i++)
        sum += __builtin_popcountl(counts[i]);

    return ((unsigned int)sum);
}
#endif
#endif



/********************************DNA FUNCTIONS *****************************************************************/



// Diep:
// store per site score to nodeNumber
#if (defined(__SSE3) || defined(__AVX))

#ifdef _WIN32

static inline void storePerSiteNodeScores (partitionList * pr, int model, INT_TYPE v, unsigned int offset , int nodeNumber)
{
  PLL_ALIGN_BEGIN unsigned int counts[INTS_PER_VECTOR] PLL_ALIGN_END;
	parsimonyNumber * buf;

	int
		i,
		j;

	VECTOR_STORE((CAST)counts, v);

  // int sum =0;

  // for(i = 0; i < INTS_PER_VECTOR; i++)
  //   sum += __builtin_popcount(counts[i]);
  // cout<<sum<<"\n";

	int partialParsLength = pr->partitionData[model]->parsimonyLength * PLL_PCF;
	int nodeStart = partialParsLength * nodeNumber;
	int nodeStartPlusOffset = nodeStart + offset * PLL_PCF;
	for (i = 0; i < INTS_PER_VECTOR; ++i){
		buf = &(pr->partitionData[model]->perSitePartialPars[nodeStartPlusOffset]);
		nodeStartPlusOffset += 32;
//		buf = &(pr->partitionData[model]->perSitePartialPars[nodeStart + offset * PLL_PCF + i * ULINT_SIZE]); // Diep's
//		buf = &(pr->partitionData[model]->perSitePartialPars[nodeStart + offset * PLL_PCF + i]); // Tomas's code
		for (j = 0; j < 32; ++ j) {
			buf[j] += ((counts[i] >> j) & 1);
    }
	}

}

#else

static inline void storePerSiteNodeScores (partitionList * pr, int model, INT_TYPE v, unsigned int offset , int nodeNumber)
{

    unsigned long
            counts[LONG_INTS_PER_VECTOR] __attribute__ ((aligned (PLL_BYTE_ALIGNMENT)));
    parsimonyNumber * buf;

    int
            i,
            j;

    VECTOR_STORE((CAST)counts, v);

    int partialParsLength = pr->partitionData[model]->parsimonyLength * PLL_PCF;
    int nodeStart = partialParsLength * nodeNumber;
    int nodeStartPlusOffset = nodeStart + offset * PLL_PCF;
    for (i = 0; i < LONG_INTS_PER_VECTOR; ++i){
        buf = &(pr->partitionData[model]->perSitePartialPars[nodeStartPlusOffset]);
        nodeStartPlusOffset += ULINT_SIZE;
//		buf = &(pr->partitionData[model]->perSitePartialPars[nodeStart + offset * PLL_PCF + i * ULINT_SIZE]); // Diep's
//		buf = &(pr->partitionData[model]->perSitePartialPars[nodeStart + offset * PLL_PCF + i]); // Tomas's code
        for (j = 0; j < ULINT_SIZE; ++ j)
            buf[j] += ((counts[i] >> j) & 1);
    }

}

#endif


// Diep:
// Add site scores in q and r to p
// q and r are children of p
template<class VectorClass>
void addPerSiteSubtreeScoresSIMD(partitionList *pr, int pNumber, int qNumber, int rNumber){
    assert(VectorClass::size() == INTS_PER_VECTOR);
    parsimonyNumber * pBuf, * qBuf, *rBuf;
    for(int i = 0; i < pr->numberOfPartitions; i++){
        int partialParsLength = pr->partitionData[i]->parsimonyLength * PLL_PCF;
        pBuf = &(pr->partitionData[i]->perSitePartialPars[partialParsLength * pNumber]);
        qBuf = &(pr->partitionData[i]->perSitePartialPars[partialParsLength * qNumber]);
        rBuf = &(pr->partitionData[i]->perSitePartialPars[partialParsLength * rNumber]);
        for(int k = 0; k < partialParsLength; k+= VectorClass::size()){
            VectorClass *pBufVC = (VectorClass*) &pBuf[k];
            VectorClass *qBufVC = (VectorClass*) &qBuf[k];
            VectorClass *rBufVC = (VectorClass*) &rBuf[k];
            *pBufVC += *qBufVC + *rBufVC;
        }
    }
}

// Diep:
// Add site scores in q and r to p
// q and r are children of p
static void addPerSiteSubtreeScores(partitionList *pr, int pNumber, int qNumber, int rNumber){
//	parsimonyNumber * pBuf, * qBuf, *rBuf;
//	for(int i = 0; i < pr->numberOfPartitions; i++){
//		int partialParsLength = pr->partitionData[i]->parsimonyLength * PLL_PCF;
//		pBuf = &(pr->partitionData[i]->perSitePartialPars[partialParsLength * pNumber]);
//		qBuf = &(pr->partitionData[i]->perSitePartialPars[partialParsLength * qNumber]);
//		rBuf = &(pr->partitionData[i]->perSitePartialPars[partialParsLength * rNumber]);
//		for(int k = 0; k < partialParsLength; k++)
//			pBuf[k] += qBuf[k] + rBuf[k];
//	}

#ifdef __AVX
    addPerSiteSubtreeScoresSIMD<Vec8ui>(pr, pNumber, qNumber, rNumber);
#else
    addPerSiteSubtreeScoresSIMD<Vec4ui>(pr, pNumber, qNumber, rNumber);
#endif
}


// Diep:
// Reset site scores of p
static void resetPerSiteNodeScores(partitionList *pr, int pNumber){
    parsimonyNumber * pBuf;
    for(int i = 0; i < pr->numberOfPartitions; i++){
        int partialParsLength = pr->partitionData[i]->parsimonyLength * PLL_PCF;
        pBuf = &(pr->partitionData[i]->perSitePartialPars[partialParsLength * pNumber]);
        memset(pBuf, 0, partialParsLength * sizeof(parsimonyNumber));
    }
}
#endif

static int checkerPars(pllInstance *tr, nodeptr p)
{
    int group = tr->constraintVector[p->number];

    if(isTip(p->number, tr->mxtips))
    {
        group = tr->constraintVector[p->number];
        return group;
    }
    else
    {
        if(group != -9)
            return group;

        group = checkerPars(tr, p->next->back);
        if(group != -9)
            return group;

        group = checkerPars(tr, p->next->next->back);
        if(group != -9)
            return group;

        return -9;
    }
}

static pllBoolean tipHomogeneityCheckerPars(pllInstance *tr, nodeptr p, int grouping)
{
    if(isTip(p->number, tr->mxtips))
    {
        if(tr->constraintVector[p->number] != grouping)
            return PLL_FALSE;
        else
            return PLL_TRUE;
    }
    else
    {
        return  (tipHomogeneityCheckerPars(tr, p->next->back, grouping) && tipHomogeneityCheckerPars(tr, p->next->next->back,grouping));
    }
}

static void getxnodeLocal (nodeptr p)
{
    nodeptr  s;

    if((s = p->next)->xPars || (s = s->next)->xPars)
    {
        p->xPars = s->xPars;
        s->xPars = 0;
    }

    assert(p->next->xPars || p->next->next->xPars || p->xPars);

}

static void computeTraversalInfoParsimony(nodeptr p, int *ti, int *counter, int maxTips, pllBoolean full, int perSiteScores)
{
#if (defined(__SSE3) || defined(__AVX))
    if(perSiteScores && pllCostMatrix == NULL){
        resetPerSiteNodeScores(iqtree->pllPartitions, p->number);
    }
#endif

    nodeptr
            q = p->next->back,
            r = p->next->next->back;

    if(! p->xPars)
        getxnodeLocal(p);

    if(full){
        if(q->number > maxTips)
            computeTraversalInfoParsimony(q, ti, counter, maxTips, full, perSiteScores);

        if(r->number > maxTips)
            computeTraversalInfoParsimony(r, ti, counter, maxTips, full, perSiteScores);
    }else{
        if(q->number > maxTips && !q->xPars)
            computeTraversalInfoParsimony(q, ti, counter, maxTips, full, perSiteScores);

        if(r->number > maxTips && !r->xPars)
            computeTraversalInfoParsimony(r, ti, counter, maxTips, full, perSiteScores);
    }

    ti[*counter]     = p->number;
    ti[*counter + 1] = q->number;
    ti[*counter + 2] = r->number;
    *counter = *counter + 4;
}

#if (defined(__SSE3) || defined(__AVX))


/**
 * Diep: Sankoff weighted parsimony
 * BQM: highly optimized vectorized version
 */
template<class VectorClass, class Numeric, const size_t states>
void newviewSankoffParsimonyIterativeFastSIMD(pllInstance *tr, partitionList * pr)
{

//    assert(VectorClass::size() == USHORT_PER_VECTOR);

    int model, *ti = tr->ti, count = ti[0], index;

    for(index = 4; index < count; index += 4) {
        size_t pNumber = (size_t)ti[index];
        size_t qNumber = (size_t)ti[index + 1];
        size_t rNumber = (size_t)ti[index + 2];
        // Diep: rNumber and qNumber are children of pNumber
        tr->parsimonyScore[pNumber] = 0;
        for(model = 0; model < pr->numberOfPartitions; model++)
        {
            size_t patterns = pr->partitionData[model]->parsimonyLength;
            assert(patterns % VectorClass::size() == 0);
            size_t i;

            Numeric *left  = (Numeric*)&(pr->partitionData[model]->parsVect)[(patterns * states * qNumber)];
            Numeric *right = (Numeric*)&(pr->partitionData[model]->parsVect)[(patterns * states * rNumber)];
            Numeric *cur   = (Numeric*)&(pr->partitionData[model]->parsVect)[(patterns * states * pNumber)];

            size_t x, z;

            /*
                    memory for score per node, assuming VectorClass::size()=2, and states=4 (A,C,G,T)
                    in block of size VectorClass::size()*states

                    Index  0  1  2  3  4  5  6  7  8  9  10 ...
                    Site   0  1  0  1  0  1  0  1  2  3   2 ...
                    State  A  A  C  C  G  G  T  T  A  A   C ...

                    // this is obsolete, vectorCostMatrix now store single entries
                    memory for cost matrix (vectorCostMatrix)
                    Index  0  1  2  3  4  5  6  7  8  9  10 ...
                    Entry AA AA AC AC AG AG AT AT CA CA  CC ...

            */

            VectorClass total_score = 0;

            for(i = 0; i < patterns; i+=VectorClass::size())
            {
                VectorClass cur_contrib = USHRT_MAX;
                size_t i_states = i*states;
                VectorClass *leftPtn = (VectorClass*) &left[i_states];
                VectorClass *rightPtn = (VectorClass*) &right[i_states];
                VectorClass *curPtn = (VectorClass*) &cur[i_states];
                Numeric *costPtn = (Numeric*)vectorCostMatrix;
                VectorClass value;
                for (z = 0; z < states; z++) {
                    VectorClass left_contrib = leftPtn[0] + costPtn[0];
                    VectorClass right_contrib = rightPtn[0] + costPtn[0];
                    for (x = 1; x < states; x++) {
                        value = leftPtn[x] + costPtn[x];
                        left_contrib = min(left_contrib, value);
                        value = rightPtn[x] + costPtn[x];
                        right_contrib = min(right_contrib, value);
                    }
                    costPtn += states;
                    cur_contrib = min(cur_contrib, (curPtn[z] = left_contrib + right_contrib));
                }

                //tr->parsimonyScore[pNumber] += cur_contrib * pr->partitionData[model]->informativePtnWgt[i];
                // because stepwise addition only check if this is > 0
                total_score += cur_contrib;
                // note that the true computation is, but the multiplication is slow
                // total_score += cur_contrib * VectorClass().load_a(&pr->partitionData[model]->informativePtnWgt[i]);
            }
            tr->parsimonyScore[pNumber] += horizontal_add(total_score);
        }
    }
}


static void newviewParsimonyIterativeFast(pllInstance *tr, partitionList *pr, int perSiteScores)
{
    if(pllCostMatrix) {
//        newviewSankoffParsimonyIterativeFast(tr, pr, perSiteScores);
//        return;
#ifdef __AVX
        if (globalParam->sankoff_short_int) {
            // using unsigned short
            switch (pr->partitionData[0]->states) {
            case 4:
                newviewSankoffParsimonyIterativeFastSIMD<Vec16us, parsimonyNumberShort, 4>(tr, pr);
                break;
            case 20:
                newviewSankoffParsimonyIterativeFastSIMD<Vec16us, parsimonyNumberShort, 20>(tr, pr);
                break;
            case 2:
                newviewSankoffParsimonyIterativeFastSIMD<Vec16us, parsimonyNumberShort, 2>(tr, pr);
                break;
            case 32:
                newviewSankoffParsimonyIterativeFastSIMD<Vec16us, parsimonyNumberShort, 32>(tr, pr);
                break;
            default:
                cerr << "Unsupported" << endl;
                exit(EXIT_FAILURE);
            }
        } else {
            // using unsigned int
            switch (pr->partitionData[0]->states) {
            case 4:
                newviewSankoffParsimonyIterativeFastSIMD<Vec8ui, parsimonyNumber, 4>(tr, pr);
                break;
            case 20:
                newviewSankoffParsimonyIterativeFastSIMD<Vec8ui, parsimonyNumber, 20>(tr, pr);
                break;
            case 2:
                newviewSankoffParsimonyIterativeFastSIMD<Vec8ui, parsimonyNumber, 2>(tr, pr);
                break;
            case 32:
                newviewSankoffParsimonyIterativeFastSIMD<Vec8ui, parsimonyNumber, 32>(tr, pr);
                break;
            default:
                cerr << "Unsupported" << endl;
                exit(EXIT_FAILURE);
            }
        }
#else // SSE code
        if (globalParam->sankoff_short_int) {
            // using unsigned short
            switch (pr->partitionData[0]->states) {
                case 4:
                    newviewSankoffParsimonyIterativeFastSIMD<Vec8us, parsimonyNumberShort, 4>(tr, pr);
                    break;
                case 20:
                    newviewSankoffParsimonyIterativeFastSIMD<Vec8us, parsimonyNumberShort, 20>(tr, pr);
                    break;
                case 2:
                    newviewSankoffParsimonyIterativeFastSIMD<Vec8us, parsimonyNumberShort, 2>(tr, pr);
                    break;
                case 32:
                    newviewSankoffParsimonyIterativeFastSIMD<Vec8us, parsimonyNumberShort, 32>(tr, pr);
                    break;
                default:
                    cerr << "Unsupported" << endl;
                    exit(EXIT_FAILURE);
            }
        } else {
            // using unsigned int
            switch (pr->partitionData[0]->states) {
                case 4:
                    newviewSankoffParsimonyIterativeFastSIMD<Vec4ui, parsimonyNumber, 4>(tr, pr);
                    break;
                case 20:
                    newviewSankoffParsimonyIterativeFastSIMD<Vec4ui, parsimonyNumber, 20>(tr, pr);
                    break;
                case 2:
                    newviewSankoffParsimonyIterativeFastSIMD<Vec4ui, parsimonyNumber, 2>(tr, pr);
                    break;
                case 32:
                    newviewSankoffParsimonyIterativeFastSIMD<Vec4ui, parsimonyNumber, 32>(tr, pr);
                    break;
                default:
                    cerr << "Unsupported" << endl;
                    exit(EXIT_FAILURE);
            }
        }
#endif
        return;
    }

    INT_TYPE
            allOne = SET_ALL_BITS_ONE;

    int
            model,
            *ti = tr->ti,
            count = ti[0],
            index;

    for(index = 4; index < count; index += 4)
    {
        unsigned int
                totalScore = 0;

        size_t
                pNumber = (size_t)ti[index],
                qNumber = (size_t)ti[index + 1],
                rNumber = (size_t)ti[index + 2];

        if(perSiteScores){
            if(qNumber <= tr->mxtips) resetPerSiteNodeScores(pr, qNumber);
            if(rNumber <= tr->mxtips) resetPerSiteNodeScores(pr, rNumber);
        }

        for(model = 0; model < pr->numberOfPartitions; model++)
        {
            size_t
                    k,
                    states = pr->partitionData[model]->states,
                    width = pr->partitionData[model]->parsimonyLength;

            unsigned int
                    i;

            switch(states)
            {
                case 2:
                {
                    parsimonyNumber
                            *left[2],
                            *right[2],
                            *cur[2];

                    for(k = 0; k < 2; k++)
                    {
                        left[k]  = &(pr->partitionData[model]->parsVect[(width * 2 * qNumber) + width * k]);
                        right[k] = &(pr->partitionData[model]->parsVect[(width * 2 * rNumber) + width * k]);
                        cur[k]  = &(pr->partitionData[model]->parsVect[(width * 2 * pNumber) + width * k]);
                    }

                    for(i = 0; i < width; i += INTS_PER_VECTOR)
                    {
                        INT_TYPE
                                s_r, s_l, v_N,
                                l_A, l_C,
                                v_A, v_C;

                        s_l = VECTOR_LOAD((CAST)(&left[0][i]));
                        s_r = VECTOR_LOAD((CAST)(&right[0][i]));
                        l_A = VECTOR_BIT_AND(s_l, s_r);
                        v_A = VECTOR_BIT_OR(s_l, s_r);

                        s_l = VECTOR_LOAD((CAST)(&left[1][i]));
                        s_r = VECTOR_LOAD((CAST)(&right[1][i]));
                        l_C = VECTOR_BIT_AND(s_l, s_r);
                        v_C = VECTOR_BIT_OR(s_l, s_r);

                        v_N = VECTOR_BIT_OR(l_A, l_C);

                        VECTOR_STORE((CAST)(&cur[0][i]), VECTOR_BIT_OR(l_A, VECTOR_AND_NOT(v_N, v_A)));
                        VECTOR_STORE((CAST)(&cur[1][i]), VECTOR_BIT_OR(l_C, VECTOR_AND_NOT(v_N, v_C)));

                        v_N = VECTOR_AND_NOT(v_N, allOne);

                        totalScore += vectorPopcount(v_N);
                        if (perSiteScores)
                            storePerSiteNodeScores(pr, model, v_N, i, pNumber);
                    }
                }
                    break;
                case 4:
                {
                    parsimonyNumber
                            *left[4],
                            *right[4],
                            *cur[4];

                    for(k = 0; k < 4; k++)
                    {
                        left[k]  = &(pr->partitionData[model]->parsVect[(width * 4 * qNumber) + width * k]);
                        right[k] = &(pr->partitionData[model]->parsVect[(width * 4 * rNumber) + width * k]);
                        cur[k]  = &(pr->partitionData[model]->parsVect[(width * 4 * pNumber) + width * k]);
                    }

                    for(i = 0; i < width; i += INTS_PER_VECTOR)
                    {
                        INT_TYPE
                                s_r, s_l, v_N,
                                l_A, l_C, l_G, l_T,
                                v_A, v_C, v_G, v_T;

                        s_l = VECTOR_LOAD((CAST)(&left[0][i]));
                        s_r = VECTOR_LOAD((CAST)(&right[0][i]));
                        l_A = VECTOR_BIT_AND(s_l, s_r);
                        v_A = VECTOR_BIT_OR(s_l, s_r);

                        s_l = VECTOR_LOAD((CAST)(&left[1][i]));
                        s_r = VECTOR_LOAD((CAST)(&right[1][i]));
                        l_C = VECTOR_BIT_AND(s_l, s_r);
                        v_C = VECTOR_BIT_OR(s_l, s_r);

                        s_l = VECTOR_LOAD((CAST)(&left[2][i]));
                        s_r = VECTOR_LOAD((CAST)(&right[2][i]));
                        l_G = VECTOR_BIT_AND(s_l, s_r);
                        v_G = VECTOR_BIT_OR(s_l, s_r);

                        s_l = VECTOR_LOAD((CAST)(&left[3][i]));
                        s_r = VECTOR_LOAD((CAST)(&right[3][i]));
                        l_T = VECTOR_BIT_AND(s_l, s_r);
                        v_T = VECTOR_BIT_OR(s_l, s_r);

                        v_N = VECTOR_BIT_OR(VECTOR_BIT_OR(l_A, l_C), VECTOR_BIT_OR(l_G, l_T));

                        VECTOR_STORE((CAST)(&cur[0][i]), VECTOR_BIT_OR(l_A, VECTOR_AND_NOT(v_N, v_A)));
                        VECTOR_STORE((CAST)(&cur[1][i]), VECTOR_BIT_OR(l_C, VECTOR_AND_NOT(v_N, v_C)));
                        VECTOR_STORE((CAST)(&cur[2][i]), VECTOR_BIT_OR(l_G, VECTOR_AND_NOT(v_N, v_G)));
                        VECTOR_STORE((CAST)(&cur[3][i]), VECTOR_BIT_OR(l_T, VECTOR_AND_NOT(v_N, v_T)));

                        v_N = VECTOR_AND_NOT(v_N, allOne);

                        totalScore += vectorPopcount(v_N);
                        if (perSiteScores)
                            storePerSiteNodeScores(pr, model, v_N, i, pNumber);
                    }
                }
                    break;
                case 20:
                {
                    parsimonyNumber
                            *left[20],
                            *right[20],
                            *cur[20];

                    for(k = 0; k < 20; k++)
                    {
                        left[k]  = &(pr->partitionData[model]->parsVect[(width * 20 * qNumber) + width * k]);
                        right[k] = &(pr->partitionData[model]->parsVect[(width * 20 * rNumber) + width * k]);
                        cur[k]  = &(pr->partitionData[model]->parsVect[(width * 20 * pNumber) + width * k]);
                    }

                    for(i = 0; i < width; i += INTS_PER_VECTOR)
                    {
                        size_t j;

                        INT_TYPE
                                s_r, s_l,
                                v_N = SET_ALL_BITS_ZERO,
                                l_A[20],
                                v_A[20];

                        for(j = 0; j < 20; j++)
                        {
                            s_l = VECTOR_LOAD((CAST)(&left[j][i]));
                            s_r = VECTOR_LOAD((CAST)(&right[j][i]));
                            l_A[j] = VECTOR_BIT_AND(s_l, s_r);
                            v_A[j] = VECTOR_BIT_OR(s_l, s_r);

                            v_N = VECTOR_BIT_OR(v_N, l_A[j]);
                        }

                        for(j = 0; j < 20; j++)
                            VECTOR_STORE((CAST)(&cur[j][i]), VECTOR_BIT_OR(l_A[j], VECTOR_AND_NOT(v_N, v_A[j])));

                        v_N = VECTOR_AND_NOT(v_N, allOne);

                        totalScore += vectorPopcount(v_N);
                        if (perSiteScores)
                            storePerSiteNodeScores(pr, model, v_N, i, pNumber);
                    }
                }
                    break;
                default:

                {
                    parsimonyNumber
                            *left[32],
                            *right[32],
                            *cur[32];

                    assert(states <= 32);

                    for(k = 0; k < states; k++)
                    {
                        left[k]  = &(pr->partitionData[model]->parsVect[(width * states * qNumber) + width * k]);
                        right[k] = &(pr->partitionData[model]->parsVect[(width * states * rNumber) + width * k]);
                        cur[k]  = &(pr->partitionData[model]->parsVect[(width * states * pNumber) + width * k]);
                    }

                    for(i = 0; i < width; i += INTS_PER_VECTOR)
                    {
                        size_t j;

                        INT_TYPE
                                s_r, s_l,
                                v_N = SET_ALL_BITS_ZERO,
                                l_A[32],
                                v_A[32];

                        for(j = 0; j < states; j++)
                        {
                            s_l = VECTOR_LOAD((CAST)(&left[j][i]));
                            s_r = VECTOR_LOAD((CAST)(&right[j][i]));
                            l_A[j] = VECTOR_BIT_AND(s_l, s_r);
                            v_A[j] = VECTOR_BIT_OR(s_l, s_r);

                            v_N = VECTOR_BIT_OR(v_N, l_A[j]);
                        }

                        for(j = 0; j < states; j++)
                            VECTOR_STORE((CAST)(&cur[j][i]), VECTOR_BIT_OR(l_A[j], VECTOR_AND_NOT(v_N, v_A[j])));

                        v_N = VECTOR_AND_NOT(v_N, allOne);

                        totalScore += vectorPopcount(v_N);
                        if (perSiteScores)
                            storePerSiteNodeScores(pr, model, v_N, i, pNumber);
                    }
                }
            }
        }

        tr->parsimonyScore[pNumber] = totalScore + tr->parsimonyScore[rNumber] + tr->parsimonyScore[qNumber];
        if (perSiteScores)
            addPerSiteSubtreeScores(pr, pNumber, qNumber, rNumber); // Diep: add rNumber and qNumber to pNumber
    }
}

template <class VectorClass, class Numeric, const size_t states, const bool BY_PATTERN>
parsimonyNumber evaluateSankoffParsimonyIterativeFastSIMD(pllInstance *tr, partitionList * pr, int perSiteScores)
{
    size_t pNumber = (size_t)tr->ti[1];
    size_t qNumber = (size_t)tr->ti[2];

    int model;

    uint32_t total_sum = 0;

    if(tr->ti[0] > 4)
        newviewParsimonyIterativeFast(tr, pr, perSiteScores);

    for(model = 0; model < pr->numberOfPartitions; model++)
    {
        size_t patterns  = pr->partitionData[model]->parsimonyLength;
        size_t i;
        Numeric *left  = (Numeric*)&(pr->partitionData[model]->parsVect)[(patterns * states * qNumber)];
        Numeric *right = (Numeric*)&(pr->partitionData[model]->parsVect)[(patterns * states * pNumber)];
        size_t x, y, seg;

        Numeric *ptnWgt = (Numeric*)pr->partitionData[model]->informativePtnWgt;
        Numeric *ptnScore = (Numeric*)pr->partitionData[model]->informativePtnScore;

        for (seg = 0; seg < pllRepsSegments; seg++) {
            VectorClass sum(0);
            size_t lower = (seg == 0) ? 0 : pllSegmentUpper[seg-1];
            size_t upper = pllSegmentUpper[seg];
            for(i = lower; i < upper; i+=VectorClass::size()){

                size_t i_states = i*states;
                VectorClass *leftPtn = (VectorClass*) &left[i_states];
                VectorClass *rightPtn = (VectorClass*) &right[i_states];
                VectorClass best_score = USHRT_MAX;
                Numeric *costRow = (Numeric*)vectorCostMatrix;

                for (x = 0; x < states; x++) {
                    VectorClass this_best_score = costRow[0] + rightPtn[0];
                    for (y = 1; y < states; y++) {
                        VectorClass value = costRow[y] + rightPtn[y];
                        this_best_score = min(this_best_score, value);
                    }
                    this_best_score += leftPtn[x];
                    best_score = min(best_score, this_best_score);
                    costRow += states;
                }

                // add weight here because weighted computation is based on pattern
                // sum += best_score * (size_t)tr->aliaswgt[i]; // wrong (because aliaswgt is for all patterns, not just informative pattern)
                if(perSiteScores) {
                    best_score.store_a(&ptnScore[i]);
                } else {
                    // TODO without having to store per site score AND finished a block of patterns
                    // then use the lower-bound to stop early
                    // if current_score + lower_bound_remaining > best_score then
                    //     return current_score + lower_bound_remaining
                }

                if (BY_PATTERN)
                    sum += best_score * VectorClass().load_a(&ptnWgt[i]);
                else
                    sum += best_score;

                // if(sum >= bestScore)
                // 		return sum;
            }

            total_sum += horizontal_add(sum);

            // Diep: IMPORTANT! since the pllRemainderLowerBounds is computed for full ntaxa
            // the following must be disabled during stepwise addition
            if((!doing_stepwise_addition) && (!perSiteScores) && (seg < pllRepsSegments - 1)){
                parsimonyNumber est_score = total_sum + pllRemainderLowerBounds[seg];
                if(est_score > tr->bestParsimony){
                    return est_score;
                }
            }
        }
    }

    return total_sum;
}



static unsigned int evaluateParsimonyIterativeFast(pllInstance *tr, partitionList *pr, int perSiteScores)
{
    if(pllCostMatrix) {
//        return evaluateSankoffParsimonyIterativeFast(tr, pr, perSiteScores);
#ifdef __AVX
        if (globalParam->sankoff_short_int) {
            switch (pr->partitionData[0]->states) {
            case 4:
                return evaluateSankoffParsimonyIterativeFastSIMD<Vec16us, parsimonyNumberShort, 4,true>(tr, pr, perSiteScores);
            case 20:
                return evaluateSankoffParsimonyIterativeFastSIMD<Vec16us, parsimonyNumberShort, 20,true>(tr, pr, perSiteScores);
            case 2:
                return evaluateSankoffParsimonyIterativeFastSIMD<Vec16us, parsimonyNumberShort, 2,true>(tr, pr, perSiteScores);
            case 32:
                return evaluateSankoffParsimonyIterativeFastSIMD<Vec16us, parsimonyNumberShort, 32,true>(tr, pr, perSiteScores);
            default:
                cerr << "Unsupported" << endl;
                exit(EXIT_FAILURE);
            }
        } else {
            switch (pr->partitionData[0]->states) {
            case 4:
                return evaluateSankoffParsimonyIterativeFastSIMD<Vec8ui, parsimonyNumber, 4,true>(tr, pr, perSiteScores);
            case 20:
                return evaluateSankoffParsimonyIterativeFastSIMD<Vec8ui, parsimonyNumber, 20,true>(tr, pr, perSiteScores);
            case 2:
                return evaluateSankoffParsimonyIterativeFastSIMD<Vec8ui, parsimonyNumber, 2,true>(tr, pr, perSiteScores);
            case 32:
                return evaluateSankoffParsimonyIterativeFastSIMD<Vec8ui, parsimonyNumber, 32,true>(tr, pr, perSiteScores);
            default:
                cerr << "Unsupported" << endl;
                exit(EXIT_FAILURE);
            }
        }
#else // SSE
        if (globalParam->sankoff_short_int) {
            switch (pr->partitionData[0]->states) {
                case 4:
                    return evaluateSankoffParsimonyIterativeFastSIMD<Vec8us, parsimonyNumberShort,4,true>(tr, pr, perSiteScores);
                case 20:
                    return evaluateSankoffParsimonyIterativeFastSIMD<Vec8us, parsimonyNumberShort,20,true>(tr, pr, perSiteScores);
                case 2:
                    return evaluateSankoffParsimonyIterativeFastSIMD<Vec8us, parsimonyNumberShort,2,true>(tr, pr, perSiteScores);
                case 32:
                    return evaluateSankoffParsimonyIterativeFastSIMD<Vec8us, parsimonyNumberShort,32,true>(tr, pr, perSiteScores);
                default:
                    cerr << "Unsupported" << endl;
                    exit(EXIT_FAILURE);
            }
        } else {
            switch (pr->partitionData[0]->states) {
                case 4:
                    return evaluateSankoffParsimonyIterativeFastSIMD<Vec4ui, parsimonyNumber,4,true>(tr, pr, perSiteScores);
                case 20:
                    return evaluateSankoffParsimonyIterativeFastSIMD<Vec4ui, parsimonyNumber,20,true>(tr, pr, perSiteScores);
                case 2:
                    return evaluateSankoffParsimonyIterativeFastSIMD<Vec4ui, parsimonyNumber,2,true>(tr, pr, perSiteScores);
                case 32:
                    return evaluateSankoffParsimonyIterativeFastSIMD<Vec4ui, parsimonyNumber,32,true>(tr, pr, perSiteScores);
                default:
                    cerr << "Unsupported" << endl;
                    exit(EXIT_FAILURE);
            }
        }
#endif
    }

    INT_TYPE
            allOne = SET_ALL_BITS_ONE;

    size_t
            pNumber = (size_t)tr->ti[1],
            qNumber = (size_t)tr->ti[2];

    int
            model;

    unsigned int
            bestScore = tr->bestParsimony,
            sum;

    if(tr->ti[0] > 4)
        newviewParsimonyIterativeFast(tr, pr, perSiteScores);

    sum = tr->parsimonyScore[pNumber] + tr->parsimonyScore[qNumber];

    if(perSiteScores){
        resetPerSiteNodeScores(pr, tr->start->number);
        addPerSiteSubtreeScores(pr, tr->start->number, pNumber, qNumber);
    }

    for(model = 0; model < pr->numberOfPartitions; model++)
    {
        size_t
                k,
                states = pr->partitionData[model]->states,
                width  = pr->partitionData[model]->parsimonyLength,
                i;

        switch(states)
        {
            case 2:
            {
                parsimonyNumber
                        *left[2],
                        *right[2];

                for(k = 0; k < 2; k++)
                {
                    left[k]  = &(pr->partitionData[model]->parsVect[(width * 2 * qNumber) + width * k]);
                    right[k] = &(pr->partitionData[model]->parsVect[(width * 2 * pNumber) + width * k]);
                }

                for(i = 0; i < width; i += INTS_PER_VECTOR)
                {
                    INT_TYPE
                            l_A = VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&left[0][i])), VECTOR_LOAD((CAST)(&right[0][i]))),
                            l_C = VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&left[1][i])), VECTOR_LOAD((CAST)(&right[1][i]))),
                            v_N = VECTOR_BIT_OR(l_A, l_C);

                    v_N = VECTOR_AND_NOT(v_N, allOne);

                    sum += vectorPopcount(v_N);
                    if(perSiteScores)
                        storePerSiteNodeScores(pr, model, v_N, i, tr->start->number);

//                 if(sum >= bestScore)
//                   return sum;
                }
            }
                break;
            case 4:
            {
                parsimonyNumber
                        *left[4],
                        *right[4];

                for(k = 0; k < 4; k++)
                {
                    left[k]  = &(pr->partitionData[model]->parsVect[(width * 4 * qNumber) + width * k]);
                    right[k] = &(pr->partitionData[model]->parsVect[(width * 4 * pNumber) + width * k]);
                }

                for(i = 0; i < width; i += INTS_PER_VECTOR)
                {
                    INT_TYPE
                            l_A = VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&left[0][i])), VECTOR_LOAD((CAST)(&right[0][i]))),
                            l_C = VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&left[1][i])), VECTOR_LOAD((CAST)(&right[1][i]))),
                            l_G = VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&left[2][i])), VECTOR_LOAD((CAST)(&right[2][i]))),
                            l_T = VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&left[3][i])), VECTOR_LOAD((CAST)(&right[3][i]))),
                            v_N = VECTOR_BIT_OR(VECTOR_BIT_OR(l_A, l_C), VECTOR_BIT_OR(l_G, l_T));

                    v_N = VECTOR_AND_NOT(v_N, allOne);

                    sum += vectorPopcount(v_N);
                    if(perSiteScores)
                        storePerSiteNodeScores(pr, model, v_N, i, tr->start->number);
//                 if(sum >= bestScore)
//                   return sum;
                }
            }
                break;
            case 20:
            {
                parsimonyNumber
                        *left[20],
                        *right[20];

                for(k = 0; k < 20; k++)
                {
                    left[k]  = &(pr->partitionData[model]->parsVect[(width * 20 * qNumber) + width * k]);
                    right[k] = &(pr->partitionData[model]->parsVect[(width * 20 * pNumber) + width * k]);
                }

                for(i = 0; i < width; i += INTS_PER_VECTOR)
                {
                    int
                            j;

                    INT_TYPE
                            l_A,
                            v_N = SET_ALL_BITS_ZERO;

                    for(j = 0; j < 20; j++)
                    {
                        l_A = VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&left[j][i])), VECTOR_LOAD((CAST)(&right[j][i])));
                        v_N = VECTOR_BIT_OR(l_A, v_N);
                    }

                    v_N = VECTOR_AND_NOT(v_N, allOne);

                    sum += vectorPopcount(v_N);
                    if(perSiteScores)
                        storePerSiteNodeScores(pr, model, v_N, i, tr->start->number);
//                  if(sum >= bestScore)
//                    return sum;
                }
            }
                break;
            default:
            {
                parsimonyNumber
                        *left[32],
                        *right[32];

                assert(states <= 32);

                for(k = 0; k < states; k++)
                {
                    left[k]  = &(pr->partitionData[model]->parsVect[(width * states * qNumber) + width * k]);
                    right[k] = &(pr->partitionData[model]->parsVect[(width * states * pNumber) + width * k]);
                }

                for(i = 0; i < width; i += INTS_PER_VECTOR)
                {
                    size_t
                            j;

                    INT_TYPE
                            l_A,
                            v_N = SET_ALL_BITS_ZERO;

                    for(j = 0; j < states; j++)
                    {
                        l_A = VECTOR_BIT_AND(VECTOR_LOAD((CAST)(&left[j][i])), VECTOR_LOAD((CAST)(&right[j][i])));
                        v_N = VECTOR_BIT_OR(l_A, v_N);
                    }

                    v_N = VECTOR_AND_NOT(v_N, allOne);

                    sum += vectorPopcount(v_N);
                    if(perSiteScores)
                        storePerSiteNodeScores(pr, model, v_N, i, tr->start->number);
//                 if(sum >= bestScore)
//                   return sum;
                }
            }
        }
    }

    return sum;
}


#else
/**
 * Diep: Sankoff weighted parsimony
 * The unvectorized version
 */
static void newviewSankoffParsimonyIterativeFast(pllInstance *tr, partitionList * pr, int perSiteScores)
{
//	cout << "newviewSankoffParsimonyIterativeFast...";
  int
    model,
    *ti = tr->ti,
    count = ti[0],
    index;

  for(index = 4; index < count; index += 4)
    {
      unsigned int
        totalScore = 0;

      size_t
        pNumber = (size_t)ti[index],
        qNumber = (size_t)ti[index + 1],
        rNumber = (size_t)ti[index + 2];
		// Diep: rNumber and qNumber are children of pNumber
		tr->parsimonyScore[pNumber] = 0;
      for(model = 0; model < pr->numberOfPartitions; model++)
        {
          size_t
            k,
            states = pr->partitionData[model]->states,
            patterns = pr->partitionData[model]->parsimonyLength;

          unsigned int
            i;

            if(states != 2 && states != 4 && states != 20) states = 32;

				parsimonyNumber
					*left,
					*right,
					*cur;

                /*
                    memory manage for storing "partial" parsimony score
                    index     0     1     2     3    4    5    6   7 ...
                    site      0     0     0     0    1    1    1   1 ...
                    state     A     C     G     T    A    C    G   T ...
                */

                left  = &(pr->partitionData[model]->parsVect[(patterns * states * qNumber)]);
                right = &(pr->partitionData[model]->parsVect[(patterns * states * rNumber)]);
                cur  = &(pr->partitionData[model]->parsVect[(patterns * states * pNumber)]);

                /*
				for(k = 0; k < states; k++)
				{

                    // this is very inefficent

                    //    site   0   1   2 .... N  0 1 2 ... N  ...
                    //    state  A   A   A .... A  C C C ... C  ...

					left[k]  = &(pr->partitionData[model]->parsVect[(width * states * qNumber) + width * k]);
					right[k] = &(pr->partitionData[model]->parsVect[(width * states * rNumber) + width * k]);
					cur[k]  = &(pr->partitionData[model]->parsVect[(width * states * pNumber) + width * k]);
				}
                */

                /*
                                  cur
                             /         \
                            /           \
                           /             \
                        left             right
                   score_left(A,C,G,T)   score_right(A,C,G,T)

                        score_cur(z) = min_x,y { cost(z->x)+score_left(x) + cost(z->y)+score_right(y)}
                                     = left_contribution + right_contribution

                        left_contribution  =  min_x{ cost(z->x)+score_left(x)}
                        right_contribution =  min_x{ cost(z->x)+score_right(x)}

                */
//                cout << "pNumber: " << pNumber << ", qNumber: " << qNumber << ", rNumber: " << rNumber << endl;
				int x, z;

                switch (states) {
                case 4:
                    for(i = 0; i < patterns; i++)
                    {
                        // cout << "i = " << i << endl;
                        parsimonyNumber cur_contrib = UINT_MAX;
                        size_t i_states = i*4;
                        parsimonyNumber *leftPtn = &left[i_states];
                        parsimonyNumber *rightPtn = &right[i_states];
                        parsimonyNumber *curPtn = &cur[i_states];
                        parsimonyNumber *costRow = pllCostMatrix;

                        for (z = 0; z < 4; z++) {
                            parsimonyNumber left_contrib = UINT_MAX;
                            parsimonyNumber right_contrib = UINT_MAX;
                            for (x = 0; x < 4; x++)
                            {
                                // if(z == 0) cout << "left[" << x << "][i] = " << left[x][i]
                                // 	<< ", right[" << x << "][i] = " << right[x][i] << endl;
                                parsimonyNumber value = costRow[x] + leftPtn[x];
                                if (value < left_contrib)
                                    left_contrib = value;

                                value = costRow[x] + rightPtn[x];
                                if (value < right_contrib)
                                    right_contrib = value;
                            }
                            curPtn[z] = left_contrib + right_contrib;
                            if(curPtn[z] < cur_contrib) cur_contrib = curPtn[z];
                            costRow += 4;
                        }

                        // totalScore += min(cur[0][i], cur[1][i], cur[2][i], cur[3][i]);

                        tr->parsimonyScore[pNumber] += cur_contrib * pr->partitionData[model]->informativePtnWgt[i];
                        // cout << "newview: " << cur_contrib << endl;

                    }
                    break;
                default:
                    for(i = 0; i < patterns; i++)
                    {
                        // cout << "i = " << i << endl;
                        parsimonyNumber cur_contrib = UINT_MAX;
                        size_t i_states = i*states;
                        parsimonyNumber *leftPtn = &left[i_states];
                        parsimonyNumber *rightPtn = &right[i_states];
                        parsimonyNumber *curPtn = &cur[i_states];
                        parsimonyNumber *costRow = pllCostMatrix;

                        for (z = 0; z < states; z++) {
                            parsimonyNumber left_contrib = UINT_MAX;
                            parsimonyNumber right_contrib = UINT_MAX;
                            for (x = 0; x < states; x++)
                            {
                                // if(z == 0) cout << "left[" << x << "][i] = " << left[x][i]
                                // 	<< ", right[" << x << "][i] = " << right[x][i] << endl;
                                parsimonyNumber value = costRow[x] + leftPtn[x];
                                if (value < left_contrib)
                                    left_contrib = value;

                                value = costRow[x] + rightPtn[x];
                                if (value < right_contrib)
                                    right_contrib = value;
                            }
                            curPtn[z] = left_contrib + right_contrib;
                            if(curPtn[z] < cur_contrib) cur_contrib = curPtn[z];
                            costRow += states;
                        }

                        // totalScore += min(cur[0][i], cur[1][i], cur[2][i], cur[3][i]);

                        tr->parsimonyScore[pNumber] += cur_contrib * pr->partitionData[model]->informativePtnWgt[i];
                        // cout << "newview: " << cur_contrib << endl;

                    }
                    break;
                }
              }


    }
//	cout << "... DONE" << endl;
}

static void newviewParsimonyIterativeFast(pllInstance *tr, partitionList *pr, int perSiteScores)
{
	if(pllCostMatrix) return newviewSankoffParsimonyIterativeFast(tr, pr, perSiteScores);
  int
    model,
    *ti = tr->ti,
    count = ti[0],
    index;

  for(index = 4; index < count; index += 4)
    {
      unsigned int
        totalScore = 0;

      size_t
        pNumber = (size_t)ti[index],
        qNumber = (size_t)ti[index + 1],
        rNumber = (size_t)ti[index + 2];

      for(model = 0; model < pr->numberOfPartitions; model++)
        {
          size_t
            k,
            states = pr->partitionData[model]->states,
            width = pr->partitionData[model]->parsimonyLength;

          unsigned int
            i;

          switch(states)
            {
            case 2:
              {
                parsimonyNumber
                  *left[2],
                  *right[2],
                  *cur[2];

                parsimonyNumber
                   o_A,
                   o_C,
                   t_A,
                   t_C,
                   t_N;

                for(k = 0; k < 2; k++)
                  {
                    left[k]  = &(pr->partitionData[model]->parsVect[(width * 2 * qNumber) + width * k]);
                    right[k] = &(pr->partitionData[model]->parsVect[(width * 2 * rNumber) + width * k]);
                    cur[k]  = &(pr->partitionData[model]->parsVect[(width * 2 * pNumber) + width * k]);
                  }

                for(i = 0; i < width; i++)
                  {
                    t_A = left[0][i] & right[0][i];
                    t_C = left[1][i] & right[1][i];

                    o_A = left[0][i] | right[0][i];
                    o_C = left[1][i] | right[1][i];

                    t_N = ~(t_A | t_C);

                    cur[0][i] = t_A | (t_N & o_A);
                    cur[1][i] = t_C | (t_N & o_C);

                    totalScore += ((unsigned int) __builtin_popcount(t_N));
                  }
              }
              break;
            case 4:
              {
                parsimonyNumber
                  *left[4],
                  *right[4],
                  *cur[4];

                for(k = 0; k < 4; k++)
                  {
                    left[k]  = &(pr->partitionData[model]->parsVect[(width * 4 * qNumber) + width * k]);
                    right[k] = &(pr->partitionData[model]->parsVect[(width * 4 * rNumber) + width * k]);
                    cur[k]  = &(pr->partitionData[model]->parsVect[(width * 4 * pNumber) + width * k]);
                  }

                parsimonyNumber
                   o_A,
                   o_C,
                   o_G,
                   o_T,
                   t_A,
                   t_C,
                   t_G,
                   t_T,
                   t_N;

                for(i = 0; i < width; i++)
                  {
                    t_A = left[0][i] & right[0][i];
                    t_C = left[1][i] & right[1][i];
                    t_G = left[2][i] & right[2][i];
                    t_T = left[3][i] & right[3][i];

                    o_A = left[0][i] | right[0][i];
                    o_C = left[1][i] | right[1][i];
                    o_G = left[2][i] | right[2][i];
                    o_T = left[3][i] | right[3][i];

                    t_N = ~(t_A | t_C | t_G | t_T);

                    cur[0][i] = t_A | (t_N & o_A);
                    cur[1][i] = t_C | (t_N & o_C);
                    cur[2][i] = t_G | (t_N & o_G);
                    cur[3][i] = t_T | (t_N & o_T);

                    totalScore += ((unsigned int) __builtin_popcount(t_N));
                  }
              }
              break;
            case 20:
              {
                parsimonyNumber
                  *left[20],
                  *right[20],
                  *cur[20];

                parsimonyNumber
                  o_A[20],
                  t_A[20],
                  t_N;

                for(k = 0; k < 20; k++)
                  {
                    left[k]  = &(pr->partitionData[model]->parsVect[(width * 20 * qNumber) + width * k]);
                    right[k] = &(pr->partitionData[model]->parsVect[(width * 20 * rNumber) + width * k]);
                    cur[k]  = &(pr->partitionData[model]->parsVect[(width * 20 * pNumber) + width * k]);
                  }

                for(i = 0; i < width; i++)
                  {
                    size_t k;

                    t_N = 0;

                    for(k = 0; k < 20; k++)
                      {
                        t_A[k] = left[k][i] & right[k][i];
                        o_A[k] = left[k][i] | right[k][i];
                        t_N = t_N | t_A[k];
                      }

                    t_N = ~t_N;

                    for(k = 0; k < 20; k++)
                      cur[k][i] = t_A[k] | (t_N & o_A[k]);

                    totalScore += ((unsigned int) __builtin_popcount(t_N));
                  }
              }
              break;
            default:
              {
                parsimonyNumber
                  *left[32],
                  *right[32],
                  *cur[32];

                parsimonyNumber
                  o_A[32],
                  t_A[32],
                  t_N;

                assert(states <= 32);

                for(k = 0; k < states; k++)
                  {
                    left[k]  = &(pr->partitionData[model]->parsVect[(width * states * qNumber) + width * k]);
                    right[k] = &(pr->partitionData[model]->parsVect[(width * states * rNumber) + width * k]);
                    cur[k]  = &(pr->partitionData[model]->parsVect[(width * states * pNumber) + width * k]);
                  }

                for(i = 0; i < width; i++)
                  {
                    t_N = 0;

                    for(k = 0; k < states; k++)
                      {
                        t_A[k] = left[k][i] & right[k][i];
                        o_A[k] = left[k][i] | right[k][i];
                        t_N = t_N | t_A[k];
                      }

                    t_N = ~t_N;

                    for(k = 0; k < states; k++)
                      cur[k][i] = t_A[k] | (t_N & o_A[k]);

                    totalScore += ((unsigned int) __builtin_popcount(t_N));
                  }
              }
            }
        }

      tr->parsimonyScore[pNumber] = totalScore + tr->parsimonyScore[rNumber] + tr->parsimonyScore[qNumber];
    }
}

static unsigned int evaluateSankoffParsimonyIterativeFast(pllInstance *tr, partitionList * pr, int perSiteScores)
{
//	cout << "evaluateSankoffParsimonyIterativeFast ...";
  size_t
    pNumber = (size_t)tr->ti[1],
    qNumber = (size_t)tr->ti[2];

  int
    model;

  unsigned int
    bestScore = tr->bestParsimony,
    sum;

  if(tr->ti[0] > 4)
    newviewParsimonyIterativeFast(tr, pr, perSiteScores);

//  sum = tr->parsimonyScore[pNumber] + tr->parsimonyScore[qNumber];
	sum = 0;

	for(model = 0; model < pr->numberOfPartitions; model++)
	{
		size_t
			k,
			states = pr->partitionData[model]->states,
			patterns  = pr->partitionData[model]->parsimonyLength,
			i;

		if(states != 2 && states != 4 && states != 20) states = 32;

		parsimonyNumber
			*left,
			*right;

			left  = &(pr->partitionData[model]->parsVect[(patterns * states * qNumber)]);
			right = &(pr->partitionData[model]->parsVect[(patterns * states * pNumber)]);
        /*
		for(k = 0; k < states; k++)
		{
			left[k]  = &(pr->partitionData[model]->parsVect[(width * states * qNumber) + width * k]);
			right[k] = &(pr->partitionData[model]->parsVect[(width * states * pNumber) + width * k]);
		}
        */


		/*

				for each branch (left --- right), compute the score


				 left ----------------- right
			score_left(A,C,G,T)   score_right(A,C,G,T)


			score = min_x,y  { score_left(x) + cost(x-->y) + score_right(y)  }


		*/
		int x, y;

        switch (states) {
        case 4:
            for(i = 0; i < patterns; i++){
                parsimonyNumber best_score = UINT_MAX;
                size_t i_states = i*4;
                parsimonyNumber *leftPtn = &left[i_states];
                parsimonyNumber *rightPtn = &right[i_states];
                parsimonyNumber *costRow = pllCostMatrix;

                for (x = 0; x < 4; x++) {
                    parsimonyNumber this_best_score = costRow[0] + rightPtn[0];
                    for (y = 1; y < 4; y++) {
                        parsimonyNumber value = costRow[y] + rightPtn[y];
                        if (value < this_best_score) this_best_score = value;
                    }
                    this_best_score += leftPtn[x];
                    if (this_best_score < best_score)
                        best_score = this_best_score;
                    costRow += 4;
                }

                    // add weight here because weighted computation is based on pattern
                    // sum += best_score * (size_t)tr->aliaswgt[i]; // wrong (because aliaswgt is for all patterns, not just informative pattern)
                    if(perSiteScores) pr->partitionData[model]->informativePtnScore[i] = best_score;

                    sum += best_score * pr->partitionData[model]->informativePtnWgt[i];

                    // if(sum >= bestScore)
                    // 		return sum;
            }
            break;

        default:

            for(i = 0; i < patterns; i++){
                parsimonyNumber best_score = UINT_MAX;
                size_t i_states = i*states;
                parsimonyNumber *leftPtn = &left[i_states];
                parsimonyNumber *rightPtn = &right[i_states];
                parsimonyNumber *costRow = pllCostMatrix;

                for (x = 0; x < states; x++) {
                    parsimonyNumber this_best_score = costRow[0] + rightPtn[0];
                    for (y = 1; y < states; y++) {
                        parsimonyNumber value = costRow[y] + rightPtn[y];
                        if (value < this_best_score) this_best_score = value;
                    }
                    this_best_score += leftPtn[x];
                    if (this_best_score < best_score)
                        best_score = this_best_score;
                    costRow += states;
                }

                    // add weight here because weighted computation is based on pattern
                    // sum += best_score * (size_t)tr->aliaswgt[i]; // wrong (because aliaswgt is for all patterns, not just informative pattern)
                    if(perSiteScores) pr->partitionData[model]->informativePtnScore[i] = best_score;

                    sum += best_score * pr->partitionData[model]->informativePtnWgt[i];

                    // if(sum >= bestScore)
                    // 		return sum;
            }
            break;
        }
	}


  return sum;
}



static unsigned int evaluateParsimonyIterativeFast(pllInstance *tr, partitionList *pr, int perSiteScores)
{
	if(pllCostMatrix) return evaluateSankoffParsimonyIterativeFast(tr, pr, perSiteScores);

  size_t
    pNumber = (size_t)tr->ti[1],
    qNumber = (size_t)tr->ti[2];

  int
    model;

  unsigned int
    bestScore = tr->bestParsimony,
    sum;

  if(tr->ti[0] > 4)
    newviewParsimonyIterativeFast(tr, pr, perSiteScores);

  sum = tr->parsimonyScore[pNumber] + tr->parsimonyScore[qNumber];

  for(model = 0; model < pr->numberOfPartitions; model++)
    {
      size_t
        k,
        states = pr->partitionData[model]->states,
        width  = pr->partitionData[model]->parsimonyLength,
        i;

       switch(states)
         {
         case 2:
           {
             parsimonyNumber
               t_A,
               t_C,
               t_N,
               *left[2],
               *right[2];

             for(k = 0; k < 2; k++)
               {
                 left[k]  = &(pr->partitionData[model]->parsVect[(width * 2 * qNumber) + width * k]);
                 right[k] = &(pr->partitionData[model]->parsVect[(width * 2 * pNumber) + width * k]);
               }

             for(i = 0; i < width; i++)
               {
                 t_A = left[0][i] & right[0][i];
                 t_C = left[1][i] & right[1][i];

                  t_N = ~(t_A | t_C);

                  sum += ((unsigned int) __builtin_popcount(t_N));

//                 if(sum >= bestScore)
//                   return sum;
               }
           }
           break;
         case 4:
           {
             parsimonyNumber
               t_A,
               t_C,
               t_G,
               t_T,
               t_N,
               *left[4],
               *right[4];

             for(k = 0; k < 4; k++)
               {
                 left[k]  = &(pr->partitionData[model]->parsVect[(width * 4 * qNumber) + width * k]);
                 right[k] = &(pr->partitionData[model]->parsVect[(width * 4 * pNumber) + width * k]);
               }

             for(i = 0; i < width; i++)
               {
                  t_A = left[0][i] & right[0][i];
                  t_C = left[1][i] & right[1][i];
                  t_G = left[2][i] & right[2][i];
                  t_T = left[3][i] & right[3][i];

                  t_N = ~(t_A | t_C | t_G | t_T);

                  sum += ((unsigned int) __builtin_popcount(t_N));

//                 if(sum >= bestScore)
//                   return sum;
               }
           }
           break;
         case 20:
           {
             parsimonyNumber
               t_A,
               t_N,
               *left[20],
               *right[20];

              for(k = 0; k < 20; k++)
                {
                  left[k]  = &(pr->partitionData[model]->parsVect[(width * 20 * qNumber) + width * k]);
                  right[k] = &(pr->partitionData[model]->parsVect[(width * 20 * pNumber) + width * k]);
                }

              for(i = 0; i < width; i++)
                {
                  t_N = 0;

                  for(k = 0; k < 20; k++)
                    {
                      t_A = left[k][i] & right[k][i];
                      t_N = t_N | t_A;
                    }

                  t_N = ~t_N;

                  sum += ((unsigned int) __builtin_popcount(t_N));

//                  if(sum >= bestScore)
//                    return sum;
                }
           }
           break;
         default:
           {
             parsimonyNumber
               t_A,
               t_N,
               *left[32],
               *right[32];

             assert(states <= 32);

             for(k = 0; k < states; k++)
               {
                 left[k]  = &(pr->partitionData[model]->parsVect[(width * states * qNumber) + width * k]);
                 right[k] = &(pr->partitionData[model]->parsVect[(width * states * pNumber) + width * k]);
               }

             for(i = 0; i < width; i++)
               {
                 t_N = 0;

                 for(k = 0; k < states; k++)
                   {
                     t_A = left[k][i] & right[k][i];
                     t_N = t_N | t_A;
                   }

                  t_N = ~t_N;

                  sum += ((unsigned int) __builtin_popcount(t_N));

//                 if(sum >= bestScore)
//                   return sum;
               }
           }
         }
    }

  return sum;
}

#endif






static unsigned int evaluateParsimony(pllInstance *tr, partitionList *pr, nodeptr p, pllBoolean full, int perSiteScores)
{
    volatile unsigned int result;
    nodeptr q = p->back;
    int *ti = tr->ti, counter = 4;

    ti[1] = p->number;
    ti[2] = q->number;

    if(full){
        if(p->number > tr->mxtips)
            computeTraversalInfoParsimony(p, ti, &counter, tr->mxtips, full, perSiteScores);
        if(q->number > tr->mxtips)
            computeTraversalInfoParsimony(q, ti, &counter, tr->mxtips, full, perSiteScores);
    }else{
        if(p->number > tr->mxtips && !p->xPars)
            computeTraversalInfoParsimony(p, ti, &counter, tr->mxtips, full, perSiteScores);
        if(q->number > tr->mxtips && !q->xPars)
            computeTraversalInfoParsimony(q, ti, &counter, tr->mxtips, full, perSiteScores);
    }

    ti[0] = counter;

    result = evaluateParsimonyIterativeFast(tr, pr, perSiteScores);

    return result;
}


static void newviewParsimony(pllInstance *tr, partitionList *pr, nodeptr  p, int perSiteScores)
{
    if(p->number <= tr->mxtips)
        return;

    {
        int counter = 4;

        computeTraversalInfoParsimony(p, tr->ti, &counter, tr->mxtips, PLL_FALSE, perSiteScores);
        tr->ti[0] = counter;

        newviewParsimonyIterativeFast(tr, pr, perSiteScores);
    }
}

/*
 * Diep: copy new version from Tomas's code for site pars
 * Here, informative site == variant site
 * IMPORTANT: 	If this function changes the definition for 'informative site' as in the below comment
 * 				the function of compressSankoffDNA needs revising
 */
/* check whether site contains at least 2 different letters, i.e.
   whether it will generate a score */
static pllBoolean isInformative(pllInstance *tr, int dataType, int site)
{
    if(globalParam && !globalParam->sort_alignment)
        return PLL_TRUE; // because of the sync between IQTree and PLL alignment (to get correct freq of pattern)

    int
            informativeCounter = 0,
            check[256],
            j,
            undetermined = getUndetermined(dataType);

    const unsigned int
            *bitVector = getBitVector(dataType);

    unsigned char
            nucleotide;


    for(j = 0; j < 256; j++)
        check[j] = 0;

    for(j = 1; j <= tr->mxtips; j++)
    {
        nucleotide = tr->yVector[j][site];
        check[nucleotide] = 1;
        assert(bitVector[nucleotide] > 0);
    }

    for(j = 0; j < undetermined; j++)
    {
        if(check[j] > 0)
            informativeCounter++;
    }

    if(informativeCounter > 1)
        return PLL_TRUE;

    return PLL_FALSE;

}

static void determineUninformativeSites(pllInstance *tr, partitionList *pr, int *informative)
{
    int
            model,
            number = 0,
            i;

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


    for(model = 0; model < pr->numberOfPartitions; model++)
    {

        for(i = pr->partitionData[model]->lower; i < pr->partitionData[model]->upper; i++)
        {
            if(isInformative(tr, pr->partitionData[model]->dataType, i)){
                informative[i] = 1;
            }
            else
            {
                informative[i] = 0;
            }
        }
    }


    /* printf("Uninformative Patterns: %d\n", number); */
}

template<class Numeric, const int VECSIZE>
static void compressSankoffDNA(pllInstance *tr, partitionList *pr, int *informative, int perSiteScores)
{
//	cout << "Begin compressSankoffDNA()" << endl;
    size_t
            totalNodes,
            i,
            model;

    totalNodes = 2 * (size_t)tr->mxtips;


    for(model = 0; model < (size_t) pr->numberOfPartitions; model++)
    {
        size_t
                k,
                states = (size_t)pr->partitionData[model]->states,
                compressedEntries,
                compressedEntriesPadded,
                entries = 0,
                lower = pr->partitionData[model]->lower,
                upper = pr->partitionData[model]->upper;

//      parsimonyNumber
//        **compressedTips = (parsimonyNumber **)rax_malloc(states * sizeof(parsimonyNumber*)),
//        *compressedValues = (parsimonyNumber *)rax_malloc(states * sizeof(parsimonyNumber));

        for(i = lower; i < upper; i++)
            if(informative[i])
                entries ++; // Diep: here,entries counts # informative pattern

        // number of informative site patterns
        compressedEntries = entries;

#if (defined(__SSE3) || defined(__AVX))
        if(compressedEntries % VECSIZE != 0)
            compressedEntriesPadded = compressedEntries + (VECSIZE - (compressedEntries % VECSIZE));
        else
            compressedEntriesPadded = compressedEntries;
#else
        compressedEntriesPadded = compressedEntries;
#endif

        // parsVect stores cost for each node by state at each pattern
        // for a certain node of DNA: ptn1_A, ptn2_A, ptn3_A,..., ptn1_C, ptn2_C, ptn3_C,...,ptn1_G, ptn2_G, ptn3_G,...,ptn1_T, ptn2_T, ptn3_T,...,
        // (not 100% sure) this is also the perSitePartialPars

        rax_posix_memalign ((void **) &(pr->partitionData[model]->parsVect), PLL_BYTE_ALIGNMENT, (size_t)compressedEntriesPadded * states * totalNodes * sizeof(parsimonyNumber));
        memset(pr->partitionData[model]->parsVect, 0, compressedEntriesPadded * states * totalNodes * sizeof(parsimonyNumber));

        //Here, without option -short_off, Numeric is 'usigned short'. So, only first half of array 'informativePtnWgt' is allocated
        //and we can not directly access this array's elements. A proposed way is creating a reference with type cast:
        //Numeric *ptnWgt = (Numeric*)pr->partitionData[model]->informativePtnWgt;
        rax_posix_memalign ((void **) &(pr->partitionData[model]->informativePtnWgt), PLL_BYTE_ALIGNMENT, (size_t)compressedEntriesPadded * sizeof(Numeric));

        memset(pr->partitionData[model]->informativePtnWgt, 0, (size_t)compressedEntriesPadded * sizeof(Numeric));

        if(perSiteScores){
            rax_posix_memalign ((void **) &(pr->partitionData[model]->informativePtnScore), PLL_BYTE_ALIGNMENT, (size_t)compressedEntriesPadded * sizeof(Numeric));
            memset(pr->partitionData[model]->informativePtnScore, 0, (size_t)compressedEntriesPadded * sizeof(Numeric));
        }

//      if (perSiteScores)
//       {
//         /* for per site parsimony score at each node */
//         rax_posix_memalign ((void **) &(pr->partitionData[model]->perSitePartialPars), PLL_BYTE_ALIGNMENT, totalNodes * (size_t)compressedEntriesPadded * PLL_PCF * sizeof (parsimonyNumber));
//         for (i = 0; i < totalNodes * (size_t)compressedEntriesPadded * PLL_PCF; ++i)
//        	 pr->partitionData[model]->perSitePartialPars[i] = 0;
//       }

        // Diep: For each leaf
        for(i = 0; i < (size_t)tr->mxtips; i++)
        {
            size_t
                    w = 0,
                    compressedIndex = 0,
                    compressedCounter = 0,
                    index = 0,
                    informativeIndex = 0;

//          for(k = 0; k < states; k++)
//            {
//              compressedTips[k] = &(pr->partitionData[model]->parsVect[(compressedEntriesPadded * states * (i + 1)) + (compressedEntriesPadded * k)]);
//              compressedValues[k] = INT_MAX; // Diep
//            }

            Numeric *tipVect = (Numeric*)&pr->partitionData[model]->parsVect[(compressedEntriesPadded * states * (i + 1))];

            Numeric *ptnWgt = (Numeric*)pr->partitionData[model]->informativePtnWgt;
            // for each informative pattern
            for(index = lower; index < (size_t)upper; index++)
            {

                if(informative[index])
                {
//                	cout << "index = " << index << endl;
                    const unsigned int
                            *bitValue = getBitVector(pr->partitionData[model]->dataType); // Diep: bitValue is for dataType

                    parsimonyNumber
                            value = bitValue[tr->yVector[i + 1][index]];

                    /*
                            memory for score per node, assuming VectorClass::size()=2, and states=4 (A,C,G,T)
                            in block of size VectorClass::size()*states

                            Index  0  1  2  3  4  5  6  7  8  9  10 ...
                            Site   0  1  0  1  0  1  0  1  2  3   2 ...
                            State  A  A  C  C  G  G  T  T  A  A   C ...

                    */

                    for(k = 0; k < states; k++)
                    {
                        if(value & mask32[k])
                            tipVect[k*VECSIZE] = 0; // Diep: if the state is present, corresponding value is set to zero
                        else
                            tipVect[k*VECSIZE] = highest_cost;
//					  compressedTips[k][informativeIndex] = compressedValues[k]; // Diep
//					  cout << "compressedValues[k]: " << compressedValues[k] << endl;
                    }
                    ptnWgt[informativeIndex] = tr->aliaswgt[index];
                    informativeIndex++;

                    tipVect += 1; // process to the next site

                    // jump to the next block
                    if (informativeIndex % VECSIZE == 0)
                        tipVect += VECSIZE*(states-1);


                }
            }

            // dummy values for the last padded entries
            for(index = informativeIndex; index < compressedEntriesPadded; index++)
            {

                for(k = 0; k < states; k++)
                {
                    tipVect[k*VECSIZE] = 0;
                }
                tipVect += 1;

            }
        }

#if (defined(__SSE3) || defined(__AVX))
        pr->partitionData[model]->parsimonyLength = compressedEntriesPadded;
#else
        pr->partitionData[model]->parsimonyLength = compressedEntries; // for unvectorized version
#endif
//	cout << "compressedEntries = " << compressedEntries << endl;
//      rax_free(compressedTips);
//      rax_free(compressedValues);
    }


    // TODO: remove this for Sankoff?

    rax_posix_memalign ((void **) &(tr->parsimonyScore), PLL_BYTE_ALIGNMENT, sizeof(unsigned int) * totalNodes);

    for(i = 0; i < totalNodes; i++)
        tr->parsimonyScore[i] = 0;

    if((!perSiteScores) && pllRepsSegments > 1){
        // compute lower-bound if not currently extracting per site score AND having > 1 segments
        pllRemainderLowerBounds = new parsimonyNumber[pllRepsSegments - 1]; // last segment does not need lower bound
        assert(iqtree != NULL);
        int partitionId = 0;
        int ptn;
        int nptn = iqtree->aln->n_informative_patterns;
        int * min_ptn_pars = new int[nptn];

        for(ptn = 0; ptn < nptn; ptn++)
            min_ptn_pars[ptn] = dynamic_cast<ParsTree *>(iqtree)->findMstScore(ptn);

        Numeric *ptnWgt = (Numeric*)pr->partitionData[partitionId]->informativePtnWgt;
        for(int seg = 0; seg < pllRepsSegments - 1; seg++){
            pllRemainderLowerBounds[seg] = 0;
            for(ptn = pllSegmentUpper[seg]; ptn < nptn; ptn++){
                pllRemainderLowerBounds[seg] += min_ptn_pars[ptn] * ptnWgt[ptn];
            }
        }

        delete [] min_ptn_pars;
    }else
        pllRemainderLowerBounds = NULL;

}


static void compressDNA(pllInstance *tr, partitionList *pr, int *informative, int perSiteScores)
{
    if(pllCostMatrix != NULL) {
        if (globalParam->sankoff_short_int)
            return compressSankoffDNA<parsimonyNumberShort, USHORT_PER_VECTOR>(tr, pr, informative, perSiteScores);
        else
            return compressSankoffDNA<parsimonyNumber, INTS_PER_VECTOR>(tr, pr, informative, perSiteScores);
    }


    size_t
            totalNodes,
            i,
            model;

    totalNodes = 2 * (size_t)tr->mxtips;



    for(model = 0; model < (size_t) pr->numberOfPartitions; model++)
    {
        size_t
                k,
                states = (size_t)pr->partitionData[model]->states,
                compressedEntries,
                compressedEntriesPadded,
                entries = 0,
                lower = pr->partitionData[model]->lower,
                upper = pr->partitionData[model]->upper;

        parsimonyNumber
                **compressedTips = (parsimonyNumber **)rax_malloc(states * sizeof(parsimonyNumber*)),
                *compressedValues = (parsimonyNumber *)rax_malloc(states * sizeof(parsimonyNumber));

        pr->partitionData[model]->numInformativePatterns = 0; // to fix score bug THAT too many uninformative sites cause out-of-bound array access

        for(i = lower; i < upper; i++)
            if(informative[i]){
                entries += (size_t)tr->aliaswgt[i];
                pr->partitionData[model]->numInformativePatterns++;
            }

        compressedEntries = entries / PLL_PCF;

        if(entries % PLL_PCF != 0)
            compressedEntries++;

#if (defined(__SSE3) || defined(__AVX))
        if(compressedEntries % INTS_PER_VECTOR != 0)
            compressedEntriesPadded = compressedEntries + (INTS_PER_VECTOR - (compressedEntries % INTS_PER_VECTOR));
        else
            compressedEntriesPadded = compressedEntries;
#else
        compressedEntriesPadded = compressedEntries;
#endif


        rax_posix_memalign ((void **) &(pr->partitionData[model]->parsVect), PLL_BYTE_ALIGNMENT, (size_t)compressedEntriesPadded * states * totalNodes * sizeof(parsimonyNumber));

        for(i = 0; i < compressedEntriesPadded * states * totalNodes; i++)
            pr->partitionData[model]->parsVect[i] = 0;

        if (perSiteScores)
        {
            /* for per site parsimony score at each node */
            rax_posix_memalign ((void **) &(pr->partitionData[model]->perSitePartialPars), PLL_BYTE_ALIGNMENT, totalNodes * (size_t)compressedEntriesPadded * PLL_PCF * sizeof (parsimonyNumber));
            for (i = 0; i < totalNodes * (size_t)compressedEntriesPadded * PLL_PCF; ++i)
                pr->partitionData[model]->perSitePartialPars[i] = 0;
        }

        for(i = 0; i < (size_t)tr->mxtips; i++)
        {
            size_t
                    w = 0,
                    compressedIndex = 0,
                    compressedCounter = 0,
                    index = 0;

            for(k = 0; k < states; k++)
            {
                compressedTips[k] = &(pr->partitionData[model]->parsVect[(compressedEntriesPadded * states * (i + 1)) + (compressedEntriesPadded * k)]);
                compressedValues[k] = 0;
            }

            for(index = lower; index < (size_t)upper; index++)
            {
                if(informative[index])
                {
                    const unsigned int
                            *bitValue = getBitVector(pr->partitionData[model]->dataType);

                    parsimonyNumber
                            value = bitValue[tr->yVector[i + 1][index]];

                    for(w = 0; w < (size_t)tr->aliaswgt[index]; w++)
                    {
                        for(k = 0; k < states; k++)
                        {
                            if(value & mask32[k])
                                compressedValues[k] |= mask32[compressedCounter];
                        }

                        compressedCounter++;

                        if(compressedCounter == PLL_PCF)
                        {
                            for(k = 0; k < states; k++)
                            {
                                compressedTips[k][compressedIndex] = compressedValues[k];
                                compressedValues[k] = 0;
                            }

                            compressedCounter = 0;
                            compressedIndex++;
                        }
                    }
                }
            }

            for(;compressedIndex < compressedEntriesPadded; compressedIndex++)
            {
                for(;compressedCounter < PLL_PCF; compressedCounter++)
                    for(k = 0; k < states; k++)
                        compressedValues[k] |= mask32[compressedCounter];

                for(k = 0; k < states; k++)
                {
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

    rax_posix_memalign ((void **) &(tr->parsimonyScore), PLL_BYTE_ALIGNMENT, sizeof(unsigned int) * totalNodes);

    for(i = 0; i < totalNodes; i++)
        tr->parsimonyScore[i] = 0;
}

static void _updateInternalPllOnRatchet(pllInstance *tr, partitionList *pr){
//	cout << "lower = " << pr->partitionData[0]->lower << ", upper = " << pr->partitionData[0]->upper << ", aln->size() = " << iqtree->aln->size() << endl;
    for(int i = 0; i < pr->numberOfPartitions; i++){
        for(int ptn = pr->partitionData[i]->lower; ptn < pr->partitionData[i]->upper; ptn++){
            tr->aliaswgt[ptn] = iqtree->aln->at(ptn).frequency;
        }
    }
}


static void _allocateParsimonyDataStructures(pllInstance *tr, partitionList *pr, int perSiteScores)
{
    int i;
    int * informative = (int *)rax_malloc(sizeof(int) * (size_t)tr->originalCrunchedLength);
    determineUninformativeSites(tr, pr, informative);

    if(pllCostMatrix){
        for(int i = 0; i < pr->numberOfPartitions; i++){
            pr->partitionData[i]->informativePtnWgt = NULL;
            pr->partitionData[i]->informativePtnScore = NULL;
        }
    }

    compressDNA(tr, pr, informative, perSiteScores);

    for(i = tr->mxtips + 1; i <= tr->mxtips + tr->mxtips - 1; i++)
    {
        nodeptr p = tr->nodep[i];

        p->xPars = 1;
        p->next->xPars = 0;
        p->next->next->xPars = 0;
    }

    tr->ti = (int*)rax_malloc(sizeof(int) * 4 * (size_t)tr->mxtips);

    rax_free(informative);
}

static void _pllFreeParsimonyDataStructures(pllInstance *tr, partitionList *pr)
{
    size_t
            model;

    if(tr->parsimonyScore != NULL){
        rax_free(tr->parsimonyScore);
        tr->parsimonyScore = NULL;
    }

    for(model = 0; model < (size_t) pr->numberOfPartitions; ++model){
        if(pr->partitionData[model]->parsVect != NULL){
            rax_free(pr->partitionData[model]->parsVect);
            pr->partitionData[model]->parsVect = NULL;
        }
        if(pr->partitionData[model]->perSitePartialPars != NULL){
            rax_free(pr->partitionData[model]->perSitePartialPars);
            pr->partitionData[model]->perSitePartialPars = NULL;
        }
    }

    if(tr->ti != NULL){
        rax_free(tr->ti);
        tr->ti = NULL;
    }
    if(pllCostMatrix){
        for(int i = 0; i < pr->numberOfPartitions; i++){
            if(pr->partitionData[i]->informativePtnWgt != NULL){
                rax_free(pr->partitionData[i]->informativePtnWgt);
                pr->partitionData[i]->informativePtnWgt = NULL;
            }
            if(pr->partitionData[i]->informativePtnScore != NULL){
                rax_free(pr->partitionData[i]->informativePtnScore);
                pr->partitionData[i]->informativePtnScore = NULL;
            }
        }
        if(pllRemainderLowerBounds){
            delete [] pllRemainderLowerBounds;
            pllRemainderLowerBounds = NULL;
        }
    }

}


void assertNode(pllInstance * tr, nodeptr p) {
    if (p->number <= tr->mxtips) {
        return;
    }
    assert(p != p->next);
    assert(p != p->next->next);
    assert(p->next != p->next->next);
}

static nodeptr duplicateTreeTopology(pllInstance * tr, nodeptr s, std::vector<nodeptr> &unused) {
    if (s->number <= tr->mxtips) {
        // leaf
        return tr->nodep[s->number];
    }
    nodeptr a = duplicateTreeTopology(tr, s->next->back, unused);
    nodeptr b = duplicateTreeTopology(tr, s->next->next->back, unused);
    assert(!unused.empty());

    nodeptr p = unused.back();
    unused.pop_back();

    hookupDefault(p->next, a);
    hookupDefault(p->next->next, b);
    p->xPars = 1;
    p->next->xPars = 0;
    p->next->next->xPars = 0;
    assertNode(tr, p);
    return p;
}

static nodeptr pruneTreeLeaves(pllInstance * tr, nodeptr s, std::vector<nodeptr> &removed, const std::vector<bool> &bts) {
    if (s->number <= tr->mxtips) {
        // leaf
        if (bts[s->number]) {
            // prune this leaf
            return nullptr;
        }
        return tr->nodep[s->number];
    }

    nodeptr a = pruneTreeLeaves(tr, s->next->back, removed, bts);
    nodeptr b = pruneTreeLeaves(tr, s->next->next->back, removed, bts);

    if (a && b) {
        hookupDefault(s->next, a);
        hookupDefault(s->next->next, b);
        return s;
    }
    removed.push_back(s);
    return a ? a : b;
}

static void retrieveTreeLeaves(pllInstance * tr, nodeptr s, std::vector<nodeptr> &leaves) {
    if (s->number <= tr->mxtips) {
        leaves.push_back(s);
        return;
    }
    retrieveTreeLeaves(tr, s->next->back, leaves);
    retrieveTreeLeaves(tr, s->next->next->back, leaves);
}

static int pllTestTreeFusing(pllInstance * tr, partitionList * pr, nodeptr p, nodeptr q, int perSiteScores) {
    nodeptr s = p->back;
    hookupDefault(q->next, p);
    hookupDefault(q->next->next, s);
    assertNode(tr, p);
    assertNode(tr, q);

    unsigned int mp = evaluateParsimony(tr, pr, q, PLL_FALSE, perSiteScores);

    // restore
    hookupDefault(p, s);

    if (mp <= tr->bestParsimony) {
        tr->bestParsimony = mp;
    }
    return mp;
}

static int pllRestoreBestFusing(pllInstance * tr, partitionList *pr, nodeptr p, nodeptr q, int perSiteScores) {
    if (p->number <= tr->mxtips) {
        return 0;
    }

    return pllTestTreeFusing(tr, pr, p->next, q, perSiteScores) == tr->bestParsimony
        || pllTestTreeFusing(tr, pr, p->next->next, q, perSiteScores) == tr->bestParsimony
        || pllRestoreBestFusing(tr, pr, p->next->back, q, perSiteScores)
        || pllRestoreBestFusing(tr, pr, p->next->next->back, q, perSiteScores);
}

static void pllFusingTraversal(pllInstance * tr, partitionList * pr, nodeptr p, nodeptr q, int perSiteScores) {
    if (p->number <= tr->mxtips) {
        return;
    }

    pllTestTreeFusing(tr, pr, p->next, q, perSiteScores);
    pllTestTreeFusing(tr, pr, p->next->next, q, perSiteScores);

    pllFusingTraversal(tr, pr, p->next->back, q, perSiteScores);
    pllFusingTraversal(tr, pr, p->next->next->back, q, perSiteScores);
}

void pllRearrangeTreeFusing(pllInstance * targetTr, pllInstance * sourceTr, partitionList * pr, nodeptr target_branch,
                            int perSiteScores, bool save_tree) {
    // get all leaf nodes
    std::vector<nodeptr> leaves;
    retrieveTreeLeaves(sourceTr, target_branch, leaves);

    if (save_tree) {
        cout << "Removed leaves\n";
        for (auto node: leaves) {
            cout << node->number << " ";
        }
        cout << "\n";
    }

    std::vector<bool> bts(targetTr->mxtips + 1);
    for (auto ptr: leaves) {
        bts[ptr->number] = 1;
    }

    std::vector<nodeptr> inner;
    nodeptr prune1 = pruneTreeLeaves(targetTr, targetTr->start->back, inner, bts);
    nodeptr prune2 = pruneTreeLeaves(targetTr, targetTr->start, inner, bts);
    if (prune1 && prune2) {
        hookupDefault(prune1, prune2);
    } else {
        nodeptr pp = prune1 ? prune1 : prune2;
        nodeptr pp_a = pp->next->back;
        nodeptr pp_b = pp->next->next->back;
        hookupDefault(pp_a, pp_b);
        inner.push_back(pp);
    }

    nodeptr p = duplicateTreeTopology(targetTr, target_branch, inner);
    assert(inner.size() == 1);  // one inner node remaining

    nodeptr q = inner.back();
    inner.pop_back();

    nodeptr s = NULL;
    for (int i = 1; i <= targetTr->mxtips; i++) {
        if (!bts[i]) {
            s = targetTr->nodep[i];
            break;
        }
    }
    assert(s != NULL);

    nodeptr t = s->back;

    hookupDefault(q->next, s);
    hookupDefault(q->next->next, t);
    hookupDefault(p, q);

    // calculate initial scores
    int counter = 4;
    computeTraversalInfoParsimony(p, targetTr->ti, &counter, targetTr->mxtips, PLL_TRUE, perSiteScores);
    computeTraversalInfoParsimony(q, targetTr->ti, &counter, targetTr->mxtips, PLL_TRUE, perSiteScores);
    targetTr->ti[0] = counter;
    newviewParsimonyIterativeFast(targetTr, pr, perSiteScores);

    // cut tree
    hookupDefault(s, t);

    if (save_tree) {
        pllRestoreBestFusing(targetTr, pr, t, q, perSiteScores);
    } else {
        // optimize
        pllFusingTraversal(targetTr, pr, t, q, perSiteScores);
    }
}

// several steps
// copy tree from X to Y
// then
void pllOptimizeTreeFusingParsimony(pllInstance * tr, partitionList * pr, pllNewickTree * btree,
                                   pllInstance * sourceTr, IQTree *_iqtree) {
    int perSiteScores = globalParam->gbo_replicates > 0;

    iqtree = _iqtree; // update pointer to IQTree

    if(globalParam->ratchet_iter >= 0 && (iqtree->on_ratchet_hclimb1 || iqtree->on_ratchet_hclimb2)){
        // oct 23: in non-ratchet iteration, allocate is not triggered
        _updateInternalPllOnRatchet(tr, pr);
        _allocateParsimonyDataStructures(tr, pr, perSiteScores);
    }else if(first_call || (iqtree && iqtree->on_opt_btree))
        _allocateParsimonyDataStructures(tr, pr, perSiteScores); // called once if not running ratchet

    if(first_call){
        first_call = false;
    }

    std::vector<nodeptr> candidates;

    for (int i = tr->mxtips + 1; i <= tr->mxtips + tr->mxtips - 2; i++) {
        if (sourceTr->nodep[i]->back->number > tr->mxtips) {
            candidates.push_back(sourceTr->nodep[i]);
        }
        if (sourceTr->nodep[i]->next->back->number > tr->mxtips) {
            candidates.push_back(sourceTr->nodep[i]->next);
        }
        if (sourceTr->nodep[i]->next->next->back->number > tr->mxtips) {
            candidates.push_back(sourceTr->nodep[i]->next->next);
        }
    }

    pllTreeInitTopologyNewick(tr, btree, PLL_FALSE);

    nodeptr best = NULL;
    tr->bestParsimony = UINT_MAX;
    tr->bestParsimony = evaluateParsimony(tr, pr, tr->start->back, PLL_TRUE, perSiteScores);

    for (auto cand: candidates) {
        // load tree from newick
        pllTreeInitTopologyNewick(tr, btree, PLL_FALSE);

        unsigned int previousScore = tr->bestParsimony;
        pllRearrangeTreeFusing(tr, sourceTr, pr, cand, perSiteScores, false);
        unsigned int newScore = tr->bestParsimony;

        if (newScore < previousScore) {
            best = cand;
        }
    }

    pllTreeInitTopologyNewick(tr, btree, PLL_FALSE);
    pllTreeToNewick(tr->tree_string, tr, pr, tr->start->back, PLL_TRUE,
                    PLL_TRUE, 0, 0, 0, PLL_SUMMARIZE_LH, 0, 0);
    auto tree_string_1 = string(tr->tree_string);
    cout << tree_string_1 << "\n";
    if (best) {
        pllRearrangeTreeFusing(tr, sourceTr, pr, best, perSiteScores, true);

        pllTreeToNewick(tr->tree_string, tr, pr, tr->start->back, PLL_TRUE,
                        PLL_TRUE, 0, 0, 0, PLL_SUMMARIZE_LH, 0, 0);
        auto tree_string_2 = string(tr->tree_string);
        cout << tree_string_2 << "\n";
        pllTreeToNewick(sourceTr->tree_string, sourceTr, pr, sourceTr->start->back, PLL_TRUE,
                        PLL_TRUE, 0, 0, 0, PLL_SUMMARIZE_LH, 0, 0);
        auto tree_string_3 = string(sourceTr->tree_string);
        cout << tree_string_3 << "\n";
    }
}
