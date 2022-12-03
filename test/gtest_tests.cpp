#include "test_includes.hpp"

std::default_random_engine generator(std::time(0));
std::uniform_real_distribution<float> float_dist(-100, 100);
std::uniform_real_distribution<double> double_dist(-100, 100);
std::uniform_int_distribution<int> int_dist(-10000000, 10000000);
std::uniform_int_distribution<int> pos_int_dist(1, 10000000);
auto random_float = std::bind(float_dist, generator);
auto random_double = std::bind(double_dist, generator);
auto random_int = std::bind(int_dist, generator);
auto random_pos_int = std::bind(pos_int_dist, generator);

#include "type_definitions.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "auxiliary_functions_test.hpp"
#include "convex_hull_test.hpp"
#include "shape_test.hpp"
#include "bitonic_sort_tests.hpp"
#include "modified_bitonic_sort_tests.hpp"
#include "quick_sort_tests.hpp"
#include "bitonic_sort_key_value_tests.hpp"
#include "test_operator_overloads.hpp"

int main(int argc, char **argv) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window =
        glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    print("Window created!");

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
