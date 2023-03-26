cmake_minimum_required(VERSION 3.20)
project(vulkan_test LANGUAGES CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)

set(ENV{VULKAN_SDK} "/Users/remi/VulkanSDK/1.3.239.0/macOS")
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_executable(testexe main.cpp)
target_include_directories(testexe
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/includes
)
target_link_libraries(testexe
    PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
)

add_custom_target(testenv
    ${CMAKE_COMMAND} -E env
    VK_ICD_FILENAMES="$ENV{VULKAN_SDK}/share/vulkan/icd.d/MoltenVK_icd.json"
    VK_LAYER_PATH="$ENV{VULKAN_SDK}/share/vulkan/explicit_layer.d"
    ./testexe
    DEPENDS testexe
)