/*
 * test.h
 *
 *  Created on: Aug 12, 2016
 *      Author: diep
 */

#ifndef SOURCE_DIRECTORY__TEST_H_
#define SOURCE_DIRECTORY__TEST_H_

#include "tools.h"

void test(Params &params);
void testSPROnUserTree(Params &params);
void testWeightedParsimony(Params &params);
void testUppassSPRCorrectness(Params &params);
void testComputeParsimonyDNA5(Params &params);
void testTreeConvertTaxaToID(Params &params);
void testRemoveDuplicateSeq(Params &params);

#endif /* SOURCE_DIRECTORY__TEST_H_ */
