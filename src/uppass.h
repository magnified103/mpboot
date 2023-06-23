/*
 * uppass.h
 *
 *  Created on: Apr 22, 2023
 *      Author: HynDuf
 */

#ifndef UPPASS_H_
#define UPPASS_H_

#include <memory>
#include <new>
#include <type_traits>

#include "iqtree.h"
#include "parsvect.h"

void _allocateParsimonyDataStructuresUppass(pllInstance *tr, partitionList *pr,
                                            int perSiteScores);
void _pllFreeParsimonyDataStructuresUppass(pllInstance *tr, partitionList *pr);

void testUppassSPR(pllInstance *tr, partitionList *pr);
void testUppassTBR(pllInstance *tr, partitionList *pr);
int pllOptimizeTbrUppass(pllInstance *tr, partitionList *pr, int mintrav,
                         int maxtrav, IQTree *_iqtree);
int pllOptimizeSprUppass(pllInstance *tr, partitionList *pr, int mintrav,
                         int maxtrav, IQTree *_iqtree);
#endif /* UPPASS_H_ */
