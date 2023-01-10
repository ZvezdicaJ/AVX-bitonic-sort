#pragma once

#include <cstdint>
#include <span>

template <typename T>
struct InternalSortParams {
    std::span<T> span;
    std::size_t firstIdx;
    std::size_t lastIdx;
};