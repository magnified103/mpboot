/*
 * uppass.h
 *
 *  Created on: Apr 22, 2023
 *      Author: HynDuf
 */

#ifndef UPPASS_H_
#define UPPASS_H_

#include <experimental/simd>

#include "iqtree.h"

template <typename T, std::size_t N, typename Tag>
class SimdTraits;

template <class Traits>
class SimdProxy;

template <typename Traits>
class SimdSiteProxy;

template <typename Traits>
class SimdParsVectProxy;

template <typename T, std::size_t N, typename Tag>
class SimdTraits {
public:
    using value_type = T;
    using simd_type = Tag;

    template <class U = T>
    constexpr static std::size_t size() { return N * sizeof(T) / sizeof(U); }
    static simd_type zeros();
    static simd_type ones();
};

template <class Traits>
class SimdProxy {
public:
    using simd_type = typename Traits::simd_type;
    using value_type = typename Traits::value_type;

    SimdProxy() = delete;

    SimdProxy(void *array) : array((value_type *)array) {}

    simd_type load();

    void store(simd_type vect);

    value_type &operator[](std::size_t pos) {
        return array[pos];
    }

    value_type operator[](std::size_t pos) const {
        return array[pos];
    }

private:
    value_type *array;
};

template <typename Traits>
class SimdSiteProxy {
public:
    using simd_type = typename Traits::simd_type;
    using value_type = typename Traits::value_type;

    SimdSiteProxy(value_type* array) : array(array) {}

    SimdProxy<Traits> operator[](std::size_t state_id) {
        return SimdProxy<Traits>(array + (state_id * Traits::size()));
    }

private:
    value_type* array;
};

template <typename Traits>
class SimdParsVectProxy {
public:
    using simd_type = typename Traits::simd_type;
    using value_type = typename Traits::value_type;

    SimdParsVectProxy(value_type* array, std::size_t states) : array(array), states(states) {}

    SimdSiteProxy<Traits> operator[](std::size_t col_id) {
        return SimdSiteProxy<Traits>(array + (states * col_id));
    }

private:
    value_type* const array;
    std::size_t const states;
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
