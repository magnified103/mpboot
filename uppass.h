/*
 * uppass.h
 *
 *  Created on: Apr 22, 2023
 *      Author: HynDuf
 */

#ifndef UPPASS_H_
#define UPPASS_H_

#include "iqtree.h"

void _allocateParsimonyDataStructuresUppass(pllInstance *tr, partitionList *pr,
                                            int perSiteScores);
void _pllFreeParsimonyDataStructuresUppass(pllInstance *tr, partitionList *pr);

void testUppassSPR(pllInstance *tr, partitionList *pr);
void testUppassTBR(pllInstance *tr, partitionList *pr);
int pllOptimizeTbrUppassFull(pllInstance *tr, partitionList *pr, int mintrav,
                         int maxtrav, IQTree *_iqtree);
int pllOptimizeTbrUppass(pllInstance *tr, partitionList *pr, int mintrav,
                         int maxtrav, IQTree *_iqtree);
#endif /* UPPASS_H_ */
