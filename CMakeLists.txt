cmake_minimum_required(VERSION 3.22.0)

project(CSC5001-N-BODY VERSION 1.0.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-Wall -fopenmp")

set(DEPENDS_LIBS
    m
    pthread
)

set(DUMP_RESULT FALSE CACHE BOOL "Dump the results of the simulation")
set(DISPLAY FALSE CACHE BOOL "Display the results of the simulation")

# COMMON SOURCES
file(GLOB_RECURSE COMMON_SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/nbody/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/nbody/*.h)

# UI SOURCES
file(GLOB_RECURSE UI_SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/ui/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/ui/*.h)

# SEQUENTIAL BRUTE FORCE SOURCES
set(SEQUENTIAL_BRUTE_FORCE_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/sequential/nbody_brute_force.c
        ${CMAKE_CURRENT_SOURCE_DIR}/sequential/nbody_brute_force.h
    )

# SEQUENTIAL BARNES HUT SOURCES
set(SEQUENTIAL_BARNES_HUT_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/sequential/nbody_barnes_hut.c
        ${CMAKE_CURRENT_SOURCE_DIR}/sequential/nbody_barnes_hut.h
    )

# MAIN SOURCES
set(MAIN_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${CMAKE_CURRENT_SOURCE_DIR}/nbody_functions.h
    )

# CONFIGURE DUMP RESULT
if (DUMP_RESULT)
    add_definitions(-DDUMP_RESULT)
endif ()

# CONFIGURE DISPLAY
if (DISPLAY)
    list(APPEND DEPENDS_LIBS
        X11
    )

    list(APPEND COMMON_SOURCE_FILES
        ${UI_SOURCE_FILES}
    )

    add_definitions(-DDISPLAY)
endif ()


# CONFIGURE AND BUILD SEQUENTIAL BRUTE FORCE VERSION
add_library(nbody_seq_brute_force_lib STATIC ${COMMON_SOURCE_FILES} ${SEQUENTIAL_BRUTE_FORCE_SOURCE_FILES})

target_include_directories(nbody_seq_brute_force_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nbody_seq_brute_force_lib ${DEPENDS_LIBS})

add_executable(nbody_seq_brute_force ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_seq_brute_force nbody_seq_brute_force_lib)
set_target_properties(nbody_seq_brute_force PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# CONFIGURE AND BUILD SEQUENTIAL BARNES HUT VERSION
add_library(nbody_seq_barnes_hut_lib STATIC ${COMMON_SOURCE_FILES} ${SEQUENTIAL_BARNES_HUT_SOURCE_FILES})

target_include_directories(nbody_seq_barnes_hut_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nbody_seq_barnes_hut_lib ${DEPENDS_LIBS})

add_executable(nbody_seq_barnes_hut ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_seq_barnes_hut nbody_seq_barnes_hut_lib)
set_target_properties(nbody_seq_barnes_hut PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)


find_package(MPI REQUIRED)

# MPI BRUTE FORCE SOURCES
set(MPI_BRUTE_FORCE_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/nbody_brute_force.c
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/nbody_brute_force.h
    )

# MPI BARNES HUT SOURCES
set(MPI_BARNES_HUT_V1_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/barnes_hut/v1/nbody_barnes_hut.c
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/barnes_hut/v1/nbody_barnes_hut.h
    )

set(MPI_BARNES_HUT_V2_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/barnes_hut/v2/nbody_barnes_hut.c
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/barnes_hut/v2/nbody_barnes_hut.h
    )

# CONFIGURE AND BUILD MPI BRUTE FORCE VERSION
add_library(nbody_mpi_brute_force_lib STATIC ${COMMON_SOURCE_FILES} ${MPI_BRUTE_FORCE_SOURCE_FILES})

target_include_directories(nbody_mpi_brute_force_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(nbody_mpi_brute_force_lib ${MPI_C_LIBRARIES} ${DEPENDS_LIBS})

add_executable(nbody_mpi_brute_force ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_mpi_brute_force nbody_mpi_brute_force_lib)
set_target_properties(nbody_mpi_brute_force PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# CONFIGURE AND BUILD MPI BARNES HUT VERSION
add_library(nbody_mpi_barnes_hut_v1_lib STATIC ${COMMON_SOURCE_FILES} ${MPI_BARNES_HUT_V1_SOURCE_FILES})

target_include_directories(nbody_mpi_barnes_hut_v1_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(nbody_mpi_barnes_hut_v1_lib ${MPI_C_LIBRARIES} ${DEPENDS_LIBS})

add_executable(nbody_mpi_barnes_hut_v1 ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_mpi_barnes_hut_v1 nbody_mpi_barnes_hut_v1_lib)
set_target_properties(nbody_mpi_barnes_hut_v1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_library(nbody_mpi_barnes_hut_v2_lib STATIC ${COMMON_SOURCE_FILES} ${MPI_BARNES_HUT_V2_SOURCE_FILES})

target_include_directories(nbody_mpi_barnes_hut_v2_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(nbody_mpi_barnes_hut_v2_lib ${MPI_C_LIBRARIES} ${DEPENDS_LIBS})

add_executable(nbody_mpi_barnes_hut_v2 ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_mpi_barnes_hut_v2 nbody_mpi_barnes_hut_v2_lib)
set_target_properties(nbody_mpi_barnes_hut_v2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)


# OPENMP-MPI BRUTE FORCE SOURCES
set(OPENMP_MPI_BRUTE_FORCE_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/nbody_brute_force.c
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/nbody_brute_force.h
    )

# OPENMP-MPI BARNES HUT SOURCES
set(OPENMP_MPI_BARNES_HUT_V1_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/barnes_hut/v1/nbody_barnes_hut.c
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/barnes_hut/v1/nbody_barnes_hut.h
    )

set(OPENMP_MPI_BARNES_HUT_V2_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/barnes_hut/v2/nbody_barnes_hut.c
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi/barnes_hut/v2/nbody_barnes_hut.h
    )

# CONFIGURE AND BUILD OPENMP-MPI BRUTE FORCE VERSION
add_library(nbody_openmp_mpi_brute_force_lib STATIC ${COMMON_SOURCE_FILES} ${OPENMP_MPI_BRUTE_FORCE_SOURCE_FILES})

target_include_directories(nbody_openmp_mpi_brute_force_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(nbody_openmp_mpi_brute_force_lib ${MPI_C_LIBRARIES} ${DEPENDS_LIBS})

add_executable(nbody_openmp_mpi_brute_force ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_openmp_mpi_brute_force nbody_openmp_mpi_brute_force_lib)
set_target_properties(nbody_openmp_mpi_brute_force PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# CONFIGURE AND BUILD OPENMP-MPI BARNES HUT VERSION
add_library(nbody_openmp_mpi_barnes_hut_v1_lib STATIC ${COMMON_SOURCE_FILES} ${OPENMP_MPI_BARNES_HUT_V1_SOURCE_FILES})

target_include_directories(nbody_openmp_mpi_barnes_hut_v1_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(nbody_openmp_mpi_barnes_hut_v1_lib ${MPI_C_LIBRARIES} ${DEPENDS_LIBS})

add_executable(nbody_openmp_mpi_barnes_hut_v1 ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_openmp_mpi_barnes_hut_v1 nbody_openmp_mpi_barnes_hut_v1_lib)
set_target_properties(nbody_openmp_mpi_barnes_hut_v1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_library(nbody_openmp_mpi_barnes_hut_v2_lib STATIC ${COMMON_SOURCE_FILES} ${OPENMP_MPI_BARNES_HUT_V2_SOURCE_FILES})

target_include_directories(nbody_openmp_mpi_barnes_hut_v2_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})
target_link_libraries(nbody_openmp_mpi_barnes_hut_v2_lib ${MPI_C_LIBRARIES} ${DEPENDS_LIBS})

add_executable(nbody_openmp_mpi_barnes_hut_v2 ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_openmp_mpi_barnes_hut_v2 nbody_openmp_mpi_barnes_hut_v2_lib)
set_target_properties(nbody_openmp_mpi_barnes_hut_v2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)


# CONFIGURE AND BUILD TESTS
enable_testing()

# VALID BRUTE FORCE IMPLEMENTATION SOURCES FOR TESTS
set(VALID_BRUTE_FORCE_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/valid_impl/nbody_brute_force.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/valid_impl/nbody_brute_force.h
    )

# VALID BARNES HUT IMPLEMENTATION SOURCES FOR TESTS
set(VALID_BARNES_HUT_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/valid_impl/nbody_barnes_hut.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/valid_impl/nbody_barnes_hut.h
    )


# BRUTE FORCE TESTS SOURCES
set(BRUTE_FORCE_TESTS_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/nbody_brute_force_test.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/valid_impl/nbody_brute_force.h
    )

# BARNES HUT TESTS SOURCES
set(BARNES_HUT_TESTS_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/nbody_barnes_hut_test.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/valid_impl/nbody_barnes_hut.h
    )

set(TEST_DEPENDS_LIBS
        check
        subunit
    )

set(TEST_PARAMETERS
        10 # number of particles
        3 # total simulation time
    )

set(IMPLEMENTATIONS_TESTS_TO_BUILD
        "seq"
    )

# CONFIGURE AND BUILD BRUTE FORCE TEST
add_library(nbody_valid_brute_force_lib STATIC ${COMMON_SOURCE_FILES} ${VALID_BRUTE_FORCE_SOURCE_FILES})

target_include_directories(nbody_valid_brute_force_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nbody_valid_brute_force_lib ${DEPENDS_LIBS})

foreach(impl ${IMPLEMENTATIONS_TESTS_TO_BUILD})
    add_executable(nbody_${impl}_brute_force_test ${BRUTE_FORCE_TESTS_SOURCE_FILES})
    target_link_libraries(nbody_${impl}_brute_force_test nbody_${impl}_brute_force_lib nbody_valid_brute_force_lib ${TEST_DEPENDS_LIBS})
    set_target_properties(nbody_${impl}_brute_force_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

add_executable(nbody_mpi_brute_force_test ${BRUTE_FORCE_TESTS_SOURCE_FILES})
target_link_libraries(nbody_mpi_brute_force_test nbody_mpi_brute_force_lib nbody_valid_brute_force_lib ${TEST_DEPENDS_LIBS})
set_target_properties(nbody_mpi_brute_force_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(nbody_openmp_mpi_brute_force_test ${BRUTE_FORCE_TESTS_SOURCE_FILES})
target_link_libraries(nbody_openmp_mpi_brute_force_test nbody_openmp_mpi_brute_force_lib nbody_valid_brute_force_lib ${TEST_DEPENDS_LIBS})
set_target_properties(nbody_openmp_mpi_brute_force_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# CONFIGURE AND BUILD BARNES HUT TEST
add_library(nbody_valid_barnes_hut_lib STATIC ${COMMON_SOURCE_FILES} ${VALID_BARNES_HUT_SOURCE_FILES})

target_include_directories(nbody_valid_barnes_hut_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nbody_valid_barnes_hut_lib ${DEPENDS_LIBS})

foreach(impl ${IMPLEMENTATIONS_TESTS_TO_BUILD})
    add_executable(nbody_${impl}_barnes_hut_test ${BARNES_HUT_TESTS_SOURCE_FILES})
    target_link_libraries(nbody_${impl}_barnes_hut_test nbody_${impl}_barnes_hut_lib nbody_valid_barnes_hut_lib ${TEST_DEPENDS_LIBS})
    set_target_properties(nbody_${impl}_barnes_hut_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

add_executable(nbody_mpi_barnes_hut_v1_test ${BARNES_HUT_TESTS_SOURCE_FILES})
target_link_libraries(nbody_mpi_barnes_hut_v1_test nbody_mpi_barnes_hut_v1_lib nbody_valid_barnes_hut_lib ${TEST_DEPENDS_LIBS})
set_target_properties(nbody_mpi_barnes_hut_v1_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(nbody_mpi_barnes_hut_v2_test ${BARNES_HUT_TESTS_SOURCE_FILES})
target_link_libraries(nbody_mpi_barnes_hut_v2_test nbody_mpi_barnes_hut_v2_lib nbody_valid_barnes_hut_lib ${TEST_DEPENDS_LIBS})
set_target_properties(nbody_mpi_barnes_hut_v2_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(nbody_openmp_mpi_barnes_hut_v1_test ${BARNES_HUT_TESTS_SOURCE_FILES})
target_link_libraries(nbody_openmp_mpi_barnes_hut_v1_test nbody_openmp_mpi_barnes_hut_v1_lib nbody_valid_barnes_hut_lib ${TEST_DEPENDS_LIBS})
set_target_properties(nbody_openmp_mpi_barnes_hut_v1_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(nbody_openmp_mpi_barnes_hut_v2_test ${BARNES_HUT_TESTS_SOURCE_FILES})
target_link_libraries(nbody_openmp_mpi_barnes_hut_v2_test nbody_openmp_mpi_barnes_hut_v2_lib nbody_valid_barnes_hut_lib ${TEST_DEPENDS_LIBS})
set_target_properties(nbody_openmp_mpi_barnes_hut_v2_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)


# CONFIGURE TEST CASES
set(IMPLEMENTATIONS_TO_TEST
        "seq"
    )

foreach(impl ${IMPLEMENTATIONS_TO_TEST})
    add_test(NAME nbody_${impl}_brute_force_test COMMAND bin/nbody_${impl}_brute_force_test ${TEST_PARAMETERS})
    add_test(NAME nbody_${impl}_barnes_hut_test COMMAND bin/nbody_${impl}_barnes_hut_test ${TEST_PARAMETERS})
endforeach()

# CONFIGURE MPI TESTS MANUALLY TO LAUNCH IT VIA MPIRUN
add_test(
    NAME nbody_mpi_brute_force_test
    COMMAND mpirun -np 5 bin/nbody_mpi_brute_force_test ${TEST_PARAMETERS})
set_property(TEST nbody_mpi_brute_force_test PROPERTY ENVIRONMENT "RDMAV_FORK_SAFE=1")

add_test(
    NAME nbody_mpi_barnes_hut_v1_test
    COMMAND mpirun -np 5 bin/nbody_mpi_barnes_hut_v1_test ${TEST_PARAMETERS})
    set_property(TEST nbody_mpi_barnes_hut_v1_test PROPERTY ENVIRONMENT "RDMAV_FORK_SAFE=1")

add_test(
    NAME nbody_mpi_barnes_hut_v2_test
    COMMAND mpirun -np 5 bin/nbody_mpi_barnes_hut_v2_test ${TEST_PARAMETERS})
    set_property(TEST nbody_mpi_barnes_hut_v2_test PROPERTY ENVIRONMENT "RDMAV_FORK_SAFE=1")


# CONFIGURE OPENMP-MPI TESTS MANUALLY TO LAUNCH IT VIA MPIRUN
add_test(
    NAME nbody_openmp_mpi_brute_force_test
    COMMAND mpirun -np 5 bin/nbody_openmp_mpi_brute_force_test ${TEST_PARAMETERS})
set_property(TEST nbody_openmp_mpi_brute_force_test PROPERTY ENVIRONMENT "RDMAV_FORK_SAFE=1")

add_test(
    NAME nbody_openmp_mpi_barnes_hut_v1_test
    COMMAND mpirun -np 5 bin/nbody_openmp_mpi_barnes_hut_v1_test ${TEST_PARAMETERS})
    set_property(TEST nbody_openmp_mpi_barnes_hut_v1_test PROPERTY ENVIRONMENT "RDMAV_FORK_SAFE=1")

add_test(
    NAME nbody_openmp_mpi_barnes_hut_v2_test
    COMMAND mpirun -np 5 bin/nbody_openmp_mpi_barnes_hut_v2_test ${TEST_PARAMETERS})
    set_property(TEST nbody_openmp_mpi_barnes_hut_v2_test PROPERTY ENVIRONMENT "RDMAV_FORK_SAFE=1")
