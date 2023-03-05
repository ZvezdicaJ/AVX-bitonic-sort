## Full bitonic sorting network implementation using AVX2.
Required c++20 compatible compile and a cpu with AVX2 support.

Building:
- cmake --preset \<preset\>
- cmake --build ./build/\<preset\>

Available presets are:
 - For g++ / clang++:
   * Debug
   * Release
 - For MSVC compiler:
   * DebugVS
   * ReleaseVS

To build tests, Google test library is required. If not in system path, user can specify path using -DGTEST_PATH=<gtest path>. Likewise, google benchmark library is required to build benchmarks. User can pass path to Google benchmark library using: -DGB_PATH=<Google benchmark path>.







