Instructions
------------

https://vulkan-tutorial.com/

Download Vulkan SDK from https://www.lunarg.com/vulkan-sdk/

Compile the shaders to SPIR-V:

    cd shaders
    ./compile.sh

Create a build folder:

    mkdir build
    cd build

Compile and run the program:

    cmake .. -DCMAKE_BUILD_TYPE=Debug
    cmake --build . -v --target testenv