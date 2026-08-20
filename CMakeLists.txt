cmake_minimum_required(VERSION 3.20)
project(genetic_algorithms LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SIMD flags
if(MSVC)
    add_compile_options(/fp:fast)
else()
    add_compile_options(-ffast-math)
endif()

# Threading
find_package(Threads REQUIRED)

# CUDA (optional) - detect via nvcc binary
set(CUDA_AVAILABLE OFF)
find_program(CUDA_NVCC_EXECUTABLE nvcc
    HINTS "$ENV{CUDA_PATH}/bin" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"
    NO_DEFAULT_PATH)
if(NOT CUDA_NVCC_EXECUTABLE)
    find_program(CUDA_NVCC_EXECUTABLE nvcc)
endif()

if(CUDA_NVCC_EXECUTABLE)
    get_filename_component(CUDA_TOOLKIT_DIR "${CUDA_NVCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_DIR "${CUDA_TOOLKIT_DIR}" DIRECTORY)
    set(CUDA_AVAILABLE ON)
    add_compile_definitions(HAS_CUDA)
    message(STATUS "CUDA found: nvcc at ${CUDA_NVCC_EXECUTABLE}")
    message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_DIR}")
else()
    message(STATUS "CUDA not found - CPU only")
endif()

# SDL2 (local)
set(SDL2_DIR "${CMAKE_SOURCE_DIR}/deps/SDL2-2.30.10")
if(EXISTS "${SDL2_DIR}/include/SDL.h")
    message(STATUS "SDL2 found at ${SDL2_DIR}")
    add_compile_definitions(HAS_SDL2)
    set(SDL2_INCLUDE_DIR "${SDL2_DIR}/include")
    set(SDL2_LIB_DIR "${SDL2_DIR}/lib/x64")
else()
    message(STATUS "SDL2 not found - TUI only")
endif()

add_executable(ga_runner src/main.cpp src/stb_truetype_impl.c)
target_include_directories(ga_runner PRIVATE include deps)
target_link_libraries(ga_runner PRIVATE Threads::Threads)

if(CUDA_AVAILABLE)
    target_include_directories(ga_runner PRIVATE "${CUDA_TOOLKIT_DIR}/include")
endif()

if(SDL2_INCLUDE_DIR)
    target_include_directories(ga_runner PRIVATE ${SDL2_INCLUDE_DIR})
    target_link_directories(ga_runner PRIVATE ${SDL2_LIB_DIR})
    target_link_libraries(ga_runner PRIVATE SDL2 SDL2main)
    # Copy DLL to output
    add_custom_command(TARGET ga_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SDL2_LIB_DIR}/SDL2.dll"
        "$<TARGET_FILE_DIR:ga_runner>/SDL2.dll")
endif()

# CUDA compilation via custom command (avoids broken MSBuild integration)
if(CUDA_AVAILABLE)
    set(CUDA_OBJ "${CMAKE_CURRENT_BINARY_DIR}/gpu_nn.obj")
    add_custom_command(
        OUTPUT ${CUDA_OBJ}
        COMMAND "${CUDA_NVCC_EXECUTABLE}"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_nn.cu"
            -o "${CUDA_OBJ}"
            --compiler-options "/O2,/fp:fast,/MD"
            -arch=sm_86 -gencode=arch=compute_86,code=sm_86
            -gencode=arch=compute_89,code=sm_89
            -gencode=arch=compute_90,code=sm_90
            -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
            -DHAS_CUDA
        DEPENDS src/gpu_nn.cu include/ga/gpu_nn.h
        COMMENT "Compiling CUDA kernel gpu_nn.cu"
    )
    add_custom_target(cuda_kernel DEPENDS ${CUDA_OBJ})
    add_dependencies(ga_runner cuda_kernel)
    target_link_libraries(ga_runner PRIVATE "${CUDA_OBJ}")
    target_link_directories(ga_runner PRIVATE "${CUDA_TOOLKIT_DIR}/lib/x64")
    target_link_libraries(ga_runner PRIVATE cudart)
    # Copy CUDA DLLs
    file(GLOB CUDA_DLLS "${CUDA_TOOLKIT_DIR}/bin/cudart64_*.dll")
    foreach(DLL ${CUDA_DLLS})
        add_custom_command(TARGET ga_runner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}"
            "$<TARGET_FILE_DIR:ga_runner>/")
    endforeach()
endif()

if(MSVC)
    target_compile_options(ga_runner PRIVATE /W4 /O2 /Oi /Ot)
    target_link_options(ga_runner PRIVATE /OPT:REF /OPT:ICF)
else()
    target_compile_options(ga_runner PRIVATE -O3 -march=native -Wall -Wextra)
endif()
