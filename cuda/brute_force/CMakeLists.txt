cmake_minimum_required(VERSION 3.24)
# set the project name
project(nbody_cuda_brute_force LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

file(GLOB CUDA_BRUTE_FORCE_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# set_source_files_properties(${COMMON_SOURCE_FILES} ${MAIN_SOURCE_FILES} PROPERTIES LANGUAGE CXX )

add_library(nbody_cuda_brute_force_lib STATIC ${COMMON_SOURCE_FILES} ${CUDA_BRUTE_FORCE_SOURCE_FILES})
set_target_properties(nbody_cuda_brute_force_lib PROPERTIES CUDA_ARCHITECTURES native)

target_include_directories(nbody_cuda_brute_force_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(nbody_cuda_brute_force_lib PUBLIC ${DEPENDS_LIBS} ${CUDA_LIBRARIES})

add_executable(nbody_cuda_brute_force ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_cuda_brute_force nbody_cuda_brute_force_lib)
set_target_properties(nbody_cuda_brute_force PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CONGIURE TESTS
set(IMPLEMENTATION_LIBS_TO_TEST ${IMPLEMENTATION_LIBS_TO_TEST} nbody_cuda_brute_force_lib PARENT_SCOPE)
set(LIB_TEST ${LIB_TEST} "RUN_SIMULATION_TEST" PARENT_SCOPE)