//
// Created on 24/10/2024.
//

#ifndef MPBOOT_TREEFUSING_H
#define MPBOOT_TREEFUSING_H

#include "iqtree.h"

void pllOptimizeTreeFusingParsimony(pllInstance * tr, partitionList * pr, pllNewickTree * btree,
                                   pllInstance * sourceTr, IQTree *_iqtree);

#endif //MPBOOT_TREEFUSING_H
