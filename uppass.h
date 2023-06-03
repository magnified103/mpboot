/*
 * uppass.h
 *
 *  Created on: Apr 22, 2023
 *      Author: HynDuf
 */

#ifndef UPPASS_H_
#define UPPASS_H_

#include <type_traits>

#include "iqtree.h"

template <typename T, std::size_t N>
struct ParsProxyTraits {
    using value_type = T;
    constexpr static std::size_t size = N;
};

template <typename Traits, std::size_t Layer = 0>
struct ParsProxy {
    using T = typename Traits::value_type;
    constexpr static std::size_t N = Traits::size;

    ParsProxy() = default;

    /**
     * Initialize a parsimony vector proxy
     * @param width
     * @param states
     * @param array
     */
    ParsProxy(std::size_t width, std::size_t states, T* array) : width(width), states(states), array(array) {}

    template <std::size_t L = Layer, std::enable_if_t<L == 0, int> = 0>
    ParsProxy<Traits, 1> operator[](std::size_t node_id) {
        return ParsProxy<Traits, 1>(width, states, array + (states * width * node_id));
    }

    template <std::size_t L = Layer, std::enable_if_t<L == 1, int> = 0>
    ParsProxy<Traits, 2> operator[](std::size_t col_id) {
        return ParsProxy<Traits, 2>(width, states, array + (states * col_id));
    }

    template <std::size_t L = Layer, std::enable_if_t<L == 2, int> = 0>
    T* operator[](std::size_t state_id) {
        return array + (N * state_id);
    }

    std::size_t width;
    std::size_t states;
    T* array;
};

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
