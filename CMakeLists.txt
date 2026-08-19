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

if(MSVC)
    target_compile_options(ga_runner PRIVATE /W4 /O2 /Oi /Ot)
    target_link_options(ga_runner PRIVATE /OPT:REF /OPT:ICF)
else()
    target_compile_options(ga_runner PRIVATE -O3 -march=native -Wall -Wextra)
endif()
