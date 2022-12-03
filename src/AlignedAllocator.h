#pragma once

#include <cstdlib>
#include <memory>
#include <limits>
#include <stdexcept>

template<typename T, std::size_t Alignment>
class AlignedAllocator {
public: 
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    template <class U> 
    struct rebind { 
		using other = AlignedAllocator<U, Alignment>; 
	};


    constexpr AlignedAllocator() noexcept = default;
    constexpr AlignedAllocator(AlignedAllocator const &) noexcept = default;
    
    template<class U, std::size_t Align>
    constexpr AlignedAllocator(AlignedAllocator<U, Align> const &) noexcept {};

    template<class U, std::size_t Align>
	AlignedAllocator& operator=(AlignedAllocator<U, Align> const&) = delete;

	constexpr ~AlignedAllocator() {}

    [[nodiscard]] constexpr T* allocate(std::size_t n) {

        constexpr std::size_t max_size = 
            std::numeric_limits<size_type>::max() / sizeof(value_type);
        
        if (n > max_size) {
            throw std::runtime_error("Too much memory requested. Allocation failed.");
        }
        
        std::size_t allocation_size_bytes = sizeof(value_type) * n;
        return static_cast<T*>(std::aligned_alloc(Alignment, allocation_size_bytes));
    }

    constexpr void deallocate(T* p, std::size_t n) {
        std::free(p);
    }

	template<typename U, std::size_t Align>
	bool operator!=(AlignedAllocator<U, Align> const &other) const noexcept{
		return false;
	}

	template<typename U, std::size_t Align>
	bool operator==(AlignedAllocator<U, Align> const &other) const noexcept {
		return true;
	}
};