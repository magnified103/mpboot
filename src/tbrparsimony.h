
/*
 * tbrparsimony.h
 *
 */

#ifndef TBRPARSIMONY_H_
#define TBRPARSIMONY_H_
// #include <sprparsimony.h>
#include "iqtree.h"

/**
 * DTH: optimize whatever tree is stored in tr by parsimony TBR
 * @param tr: the tree instance :)
 * @param partition: the data partition :)
 * @param mintrav, maxtrav are PLL limitations for TBR radius
 * @return best parsimony score found
 */
int pllOptimizeTbrParsimonyCen(pllInstance *tr, partitionList *pr, int mintrav,
                            int maxtrav, IQTree *iqtree);
int pllOptimizeTbrParsimonyFull(pllInstance *tr, partitionList *pr, IQTree *iqtree);
int pllOptimizeTbrParsimony(pllInstance *tr, partitionList *pr, int mintrav,
                            int maxtrav, IQTree *iqtree);
int pllOptimizeTbrParsimonySuperFull(pllInstance *tr, partitionList *pr, int mintrav,
                            int maxtrav, IQTree *iqtree);
int pllOptimizeTbrParsimonyMix(pllInstance *tr, partitionList *pr, int mintrav,
                            int maxtrav, IQTree *iqtree);

void pllComputeRandomizedStepwiseAdditionParsimonyTreeTBR(
    pllInstance *tr, partitionList *partitions, int tbr_mintrav,
    int tbr_maxtrav, IQTree *_iqtree);

void testTBROnUserTree(Params &params);

#endif /* TBRPARSIMONY_H_ */
