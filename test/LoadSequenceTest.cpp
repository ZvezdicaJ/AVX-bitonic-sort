#include <maskload.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <BitonicSort.h>
#include <BitonicSortCommon.h>

namespace bitonic_sort::utils {
namespace {

template <std::size_t N, std::size_t ToLoad, std::int64_t... Vals>
void loadSequenceTestHelper(LoadSequence<Vals...> const &loadSequence) {

    static_assert(N >= ToLoad);
    std::vector<std::int64_t> loadSequenceVec{Vals...};

    std::vector<std::int64_t> expectedLoadSequence(N);
    for (std::uint32_t i = 0; i < ToLoad; i++) {
        expectedLoadSequence[N - i - 1] = LOAD;
    }
    EXPECT_EQ(expectedLoadSequence.size(), loadSequenceVec.size());
    EXPECT_THAT(expectedLoadSequence, ::testing::ContainerEq(loadSequenceVec));
}

template <std::size_t N, std::size_t ToLoad>
void loadSequenceTest() {
    return loadSequenceTestHelper<N, ToLoad>(MakeLoadSequence<N, ToLoad>());
}

TEST(LoadSequenceTest, LoadSequenceTestFloat) {
    loadSequenceTest<4, 0>();
    loadSequenceTest<4, 1>();
    loadSequenceTest<4, 2>();
    loadSequenceTest<4, 3>();
    loadSequenceTest<4, 4>();

    loadSequenceTest<8, 0>();
    loadSequenceTest<8, 2>();
    loadSequenceTest<8, 4>();
    loadSequenceTest<8, 6>();
    loadSequenceTest<8, 8>();
}

TEST(LoadSequenceTest, computeBlendMaskTest) {
    EXPECT_EQ(computeBlendMask<0>(), std::uint32_t(0));
    EXPECT_EQ(computeBlendMask<1>(), std::uint32_t(0b0001));
    EXPECT_EQ(computeBlendMask<2>(), std::uint32_t(0b0011));
    EXPECT_EQ(computeBlendMask<3>(), std::uint32_t(0b0111));
    EXPECT_EQ(computeBlendMask<4>(), std::uint32_t(0b1111));
}

template <typename RegType>
void checkMask(std::uint32_t loadCount, __m256i const &mask) {
    using IntType = typename SimdReg<RegType>::IntType;
    std::vector<IntType> maskVec(SimdSize<RegType>);
    _mm256_storeu_si256((__m256i *)(maskVec.data()), mask);
    for (std::uint32_t i = 0; i < SimdSize<RegType>; i++) {
        if (i < loadCount) {
            EXPECT_EQ(maskVec[i], IntType(LOAD));
        }
    }
}

template <typename RegType>
void runMaskLoaderTest(std::vector<typename SimdReg<RegType>::ElementType> vec1) {
    auto const [reg, mask] = maskload<RegType>(std::span(vec1));
    checkMask<RegType>(vec1.size(), mask);
}

TEST(MaskLoaderTest, floatTest) {
    runMaskLoaderTest<__m256>({1.f});
    runMaskLoaderTest<__m256>({1.f, 2.f});
    runMaskLoaderTest<__m256>({1.f, 2.f, 3.f});
    runMaskLoaderTest<__m256>({1.f, 2.f, 3.f, 4.f});
    runMaskLoaderTest<__m256>({1.f, 2.f, 3.f, 4.f, 5.f});
    runMaskLoaderTest<__m256>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    runMaskLoaderTest<__m256>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
    runMaskLoaderTest<__m256>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
}

TEST(MaskLoaderTest, intTest) {
    runMaskLoaderTest<__m256i>({1});
    runMaskLoaderTest<__m256i>({1, 2});
    runMaskLoaderTest<__m256i>({1, 2, 3});
    runMaskLoaderTest<__m256i>({1, 2, 3, 4});
    runMaskLoaderTest<__m256i>({1, 2, 3, 4, 5});
    runMaskLoaderTest<__m256i>({1, 2, 3, 4, 5, 6});
    runMaskLoaderTest<__m256i>({1, 2, 3, 4, 5, 6, 7});
    runMaskLoaderTest<__m256i>({1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(MaskLoaderTest, doubleTest) {
    runMaskLoaderTest<__m256d>({1.f});
    runMaskLoaderTest<__m256d>({1.f, 2.f});
    runMaskLoaderTest<__m256d>({1.f, 2.f, 3.f});
    runMaskLoaderTest<__m256d>({1.f, 2.f, 3.f, 4.f});
}

} // namespace
} // namespace bitonic_sort::utils