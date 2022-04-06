
/*
 * tbrparsimony.h
 *
 */

#ifndef TBRPARSIMONY_H_
#define TBRPARSIMONY_H_
// #include <sprparsimony.h>
#include "iqtree.h"

/*
 * An alternative for pllComputeRandomizedStepwiseAdditionParsimonyTree
 * because the original one seems to have the wrong deallocation function
 */

void _allocateParsimonyDataStructures(pllInstance *tr, partitionList *pr, int perSiteScores);
void _pllFreeParsimonyDataStructures(pllInstance *tr, partitionList *pr);

/**
 * TBR operations
 */
int pllTbrRemoveBranch (pllInstance * tr, partitionList * pr, nodeptr p);
static int pllTbrConnectSubtrees(pllInstance * tr, nodeptr p,
                                 nodeptr q, nodeptr * freeBranch, nodeptr * pb, nodeptr * qb);


/**
 * DTH: optimize whatever tree is stored in tr by parsimony TBR
 * @param tr: the tree instance :)
 * @param partition: the data partition :)
 * @param mintrav, maxtrav are PLL limitations for TBR radius
 * @return best parsimony score found
 */
int pllOptimizeTbrParsimony(pllInstance * tr, partitionList * pr, int mintrav, int maxtrav, IQTree *iqtree);

void testTBROnUserTree(Params &params);

#endif /* TBRPARSIMONY_H_ */