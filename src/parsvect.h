#ifndef PARSVECT_H_
#define PARSVECT_H_

#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>

#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

template <typename T, std::size_t N, std::align_val_t A>
struct ParsProxyTraits {
    using value_type = T;
    constexpr static std::size_t size = N;
    constexpr static std::align_val_t alignment = A;
};

template <typename Traits, bool Alloc, std::size_t Layer>
struct BaseParsWrapper {
    template <typename T, std::align_val_t Alignment> struct aligned_delete {};
    template<typename T, std::align_val_t Alignment>
    struct aligned_delete<T[], Alignment> {
        /// Default constructor
        constexpr aligned_delete() noexcept = default;

        /** @brief Converting constructor.
         *
         * Allows conversion from a deleter for arrays of another type, such as
         * a const-qualified version of `T`.
         *
         * Conversions from types derived from `T` are not allowed because
         * it is undefined to `delete[]` an array of derived types through a
         * pointer to the base type.
         */
        template<typename U, typename = std::enable_if_t<std::is_convertible_v<U(*)[], T(*)[]>>>
        aligned_delete(const aligned_delete<U[], Alignment>&) noexcept {}

        /// Calls `::operator delete[](ptr, Alignment)`
        template<typename U>
        typename std::enable_if_t<std::is_convertible_v<U(*)[], T(*)[]>> operator()(U* ptr) const {
            static_assert(sizeof(T) > 0, "Can't delete pointer to incomplete type");
            ::operator delete[](ptr, Alignment);
        }
    };

    using T = typename Traits::value_type;
    using Ptr = std::conditional_t<Alloc, std::unique_ptr<T[], aligned_delete<T[], Traits::alignment>>, T*>;
    constexpr static std::size_t N = Traits::size;

    BaseParsWrapper() = default;

    /**
     * @brief Constructor for proxy
     *
     * @param width Optional at layer 1
     * @param states Optional at layer 2
     * @param array
     */
    template <bool A = Alloc, typename = std::enable_if_t<!A>>
    BaseParsWrapper(std::size_t width, std::size_t states, T* array) : width(width), states(states), array(array) {}

    /**
     * @brief Converting constructor.
     *
     * Allow conversion from wrapper / proxy to proxy
     */
    template <bool B, bool A = Alloc, typename = std::enable_if_t<!A>>
    BaseParsWrapper(const BaseParsWrapper<Traits, B, Layer> &pars) : width(pars.width), states(pars.states) {
        if constexpr (B) {
            array = pars.array.get();
        } else {
            array = pars.array;
        }
    }

    /**
     * @brief Constructor for wrapper at layer 0
     *
     * Automatically allocate the necessary (aligned) memory
     * @param nodes
     * @param width
     * @param states
     */
    template <bool A = Alloc, std::size_t L = Layer, typename = std::enable_if_t<A && L == 0>>
    BaseParsWrapper(std::size_t nodes, std::size_t width, std::size_t states) : width(width), states(states) {
        array = Ptr((T*)(::operator new[](sizeof(T[nodes * states * width]), Traits::alignment)));
    }

    /**
     * @brief Constructor for wrapper at layer 1
     *
     * Automatically allocate the necessary (aligned) memory
     * @param width
     * @param states
     */
    template <bool A = Alloc, std::size_t L = Layer, typename = std::enable_if_t<A && L == 1>>
    BaseParsWrapper(std::size_t width, std::size_t states) : width(width), states(states) {
        array = Ptr((T*)(::operator new[](sizeof(T[states * width]), Traits::alignment)));
    }

    /**
     * @brief Constructor for wrapper at layer 2
     *
     * Automatically allocate the necessary (aligned) memory
     * @param width
     * @param states
     */
    template <bool A = Alloc, std::size_t L = Layer, typename = std::enable_if_t<A && L == 2>>
    BaseParsWrapper(std::size_t states) : states(states) {
            array = Ptr((T*)(::operator new[](sizeof(T[N * states]), Traits::alignment)));
    }

    /**
     * @brief Access the specified proxy
     *
     * @param node_id
     * @return
     */
    template <std::size_t L = Layer, typename = std::enable_if_t<L == 0>>
    BaseParsWrapper<Traits, false, 1> operator[](std::size_t node_id) {
        return BaseParsWrapper<Traits, false, 1>(width, states, array + (states * width * node_id));
    }

    /**
     * @brief Access the specified proxy
     *
     * @param col_id
     * @return
     */
    template <std::size_t L = Layer, typename = std::enable_if_t<L == 1>>
    BaseParsWrapper<Traits, false, 2> operator[](std::size_t col_id) {
        return BaseParsWrapper<Traits, false, 2>(width, states, array + (states * col_id));
    }

    /**
     * @brief Access the specified proxy
     *
     * @param state_id
     * @return
     */
    template <std::size_t L = Layer, typename = std::enable_if_t<L == 2>>
    T* operator[](std::size_t state_id) {
        return array + (N * state_id);
    }

    std::size_t width;
    std::size_t states;
    std::conditional_t<Alloc, std::unique_ptr<T[], aligned_delete<T[], Traits::alignment>>, T*> array;
};

/**
 * @brief A proxy class
 */
template <typename Traits, std::size_t Layer = 0>
using ParsProxy = BaseParsWrapper<Traits, false, Layer>;

/**
 * @brief A wrapper class
 */
template <typename Traits, std::size_t Layer = 0>
using ParsWrapper = BaseParsWrapper<Traits, true, Layer>;

using DefaultParsProxyTraits = ParsProxyTraits<parsimonyNumber, hn::Lanes(hn::ScalableTag<unsigned int>{}), std::align_val_t{PLL_BYTE_ALIGNMENT}>;

constexpr std::size_t BLOCK_SIZE = hn::Lanes(hn::ScalableTag<unsigned int>{});

inline constexpr std::size_t table_index(std::size_t width, std::size_t states, std::size_t node_id,
                                         std::size_t state_id, std::size_t col_id) {
    return states * width * node_id + states * (col_id / BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE * state_id + (col_id % BLOCK_SIZE);
}

inline constexpr std::size_t table_index_divisible(std::size_t width, std::size_t states, std::size_t node_id,
                                                   std::size_t state_id, std::size_t col_id) {
    return states * width * node_id + states * col_id + BLOCK_SIZE * state_id;
}

inline constexpr std::size_t next_column_block(std::size_t states, std::size_t pos) {
    return pos + BLOCK_SIZE * states;
}

inline constexpr std::size_t next_column_simd(std::size_t states, std::size_t pos) {
    return next_column_block(states, pos);
}

inline constexpr std::size_t next_state(std::size_t pos, std::size_t step = 1) {
    return pos + BLOCK_SIZE * step;
}

/**
 * Tranform the table schema from (node x state x column) to the designated representation
 * @tparam T
 * @param table
 */
template <typename T>
inline void table_transform(T* table, int nodes, int states, int width) {
    std::vector<T> tmp(table, table + (nodes * states * width));
    for (int node = 0; node < nodes; node++) {
        for (int state = 0; state < states; state++) {
            for (int col = 0; col < width; col++) {
                table[table_index(width, states, node, state, col)] = tmp[node * states * width + state * width + col];
            }
        }
    }
}

#endif /* PARSVECT_H_ */
