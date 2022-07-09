/*
 * sprparsimony.h
 *
 *  Created on: Nov 6, 2014
 *      Author: diep
 */

#ifndef SPRPARSIMONY_H_
#define SPRPARSIMONY_H_

#include "iqtree.h"
#include "pllrepo/src/pll.h"

/*
 * An alternative for pllComputeRandomizedStepwiseAdditionParsimonyTree
 * because the original one seems to have the wrong deallocation function
 */
void _pllComputeRandomizedStepwiseAdditionParsimonyTree(
    pllInstance *tr, partitionList *partitions, int sprDist, IQTree *_iqtree);

template <int perSiteScores>
void _allocateParsimonyDataStructures(pllInstance *tr, partitionList *pr);
void _pllFreeParsimonyDataStructures(pllInstance *tr, partitionList *pr);

/**
 * DTH: optimize whatever tree is stored in tr by parsimony SPR
 * @param tr: the tree instance :)
 * @param partition: the data partition :)
 * @param mintrav, maxtrav are PLL limitations for SPR radius
 * @return best parsimony score found
 */
int pllOptimizeSprParsimony(pllInstance *tr, partitionList *pr, int mintrav,
                            int maxtrav, IQTree *iqtree);
int pllOptimizeSprParsimony(pllInstance *tr, partitionList *pr, int mintrav,
                            int maxtrav, IQTree *iqtree, bool spr_better);

int pllSaveCurrentTreeSprParsimony(pllInstance *tr, partitionList *pr,
                                   int cur_search_pars);

void pllComputePatternParsimony(pllInstance *tr, partitionList *pr,
                                double *ptn_npars, double *cur_npars);
void pllComputePatternParsimony(pllInstance *tr, partitionList *pr,
                                unsigned short *ptn_pars, int *cur_pars);
void pllComputePatternParsimonySlow(pllInstance *tr, partitionList *pr,
                                    double *ptn_npars,
                                    double *cur_npars); // old version

void pllComputeSiteParsimony(pllInstance *tr, partitionList *pr, int *site_pars,
                             int nsite, int *cur_pars = NULL);
void pllComputeSiteParsimony(pllInstance *tr, partitionList *pr,
                             unsigned short *site_pars, int nsite,
                             int *cur_pars = NULL);

int pllCalcMinParsScorePattern(pllInstance *tr, int dataType, int site);

// Diep: for testing site parsimony computed by PLL vs IQTree on the same tree
// this is called if params.test_site_pars == true
void testSiteParsimony(Params &params);
void testSPROnUserTree(Params &params);

void computeUserTreeParsimomy(Params &params);
void convertNewickToTnt(Params &params);
void convertNewickToNexus(Params &params);
// util function
// act as pllAlignmentRemoveDups of PLL but for sorted alignment of IQTREE
extern void pllSortedAlignmentRemoveDups(pllAlignmentData *alignmentData,
                                         partitionList *pl); /* Diep added */

// TBR uses these functions

/**
 * Diep: Sankoff weighted parsimony
 * BQM: highly optimized vectorized version
 */
template <class VectorClass, class Numeric, const size_t states>
void newviewSankoffParsimonyIterativeFastSIMD(pllInstance *tr,
                                              partitionList *pr);

template <int perSiteScores>
void _newviewParsimonyIterativeFast(pllInstance *tr, partitionList *pr);

template <class VectorClass, class Numeric, const size_t states,
          const bool BY_PATTERN>
parsimonyNumber evaluateSankoffParsimonyIterativeFastSIMD(pllInstance *tr,
                                                          partitionList *pr,
                                                          int perSiteScores);

template <int perSiteScores>
unsigned int _evaluateParsimonyIterativeFast(pllInstance *tr, partitionList *pr);

/**
 * Diep: Sankoff weighted parsimony
 * The unvectorized version
 */
void _newviewSankoffParsimonyIterativeFast(pllInstance *tr, partitionList *pr,
                                           int perSiteScores);

unsigned int _evaluateSankoffParsimonyIterativeFast(pllInstance *tr,
                                                    partitionList *pr,
                                                    int perSiteScores);

template <int perSiteScores>
unsigned int _evaluateParsimony(pllInstance *tr, partitionList *pr, nodeptr p,
                                pllBoolean full);

template <int perSiteScores>
void _newviewParsimony(pllInstance *tr, partitionList *pr, nodeptr p);
#endif /* SPRPARSIMONY_H_ */
