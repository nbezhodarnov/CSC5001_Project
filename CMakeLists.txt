cmake_minimum_required(VERSION 3.24)

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_C_STANDARD 99)

project(CSC5001-N-BODY VERSION 1.0.0)

set(CMAKE_C_FLAGS "-Wall -fopenmp")

set(DEPENDS_LIBS
    m
    pthread
)

set(DUMP_RESULT TRUE CACHE BOOL "Dump the results of the simulation")

# THIS OPTION IS NOT TESTED FOR ALL TARGETS SINCE WE RELY ON UNIT TESTS TO CHECK THE CORRECTNESS OF THE IMPLEMENTATIONS
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

# OPENMP BRUTE FORCE SOURCES
set(OPENMP_BRUTE_FORCE_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/openmp/nbody_brute_force.c
        ${CMAKE_CURRENT_SOURCE_DIR}/openmp/nbody_brute_force.h
    )

# OPENMP BARNES HUT SOURCES
set(OPENMP_BARNES_HUT_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/openmp/nbody_barnes_hut.c
        ${CMAKE_CURRENT_SOURCE_DIR}/openmp/nbody_barnes_hut.h
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

# # LIST OF VALID (SEQUENTIAL) IMPLEMENTATIONS TO TEST
# set(VALID_IMPLEMENTATION_LIBS_TO_TEST "")

# LIST OF PARALLEL IMPLEMENTATIONS TO TEST
set(IMPLEMENTATION_LIBS_TO_TEST "")

# LIST OF TESTS TO RUN FOR EACH IMPLEMENTATION
set(LIB_TEST "")

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


# CONFIGURE AND BUILD CUDA VERSIONS
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda)

# CONFIGURE AND BUILD OPENMP VERSIONS
file(GLOB OPENMP_APPROACH_FOLDER_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/openmp/*)

foreach(APPROACH ${OPENMP_APPROACH_FOLDER_LIST})
    if (NOT IS_DIRECTORY ${APPROACH})
        continue()
    endif ()

    file(GLOB VERSION_FOLDER_LIST ${APPROACH}/*)

    foreach(VERSION ${VERSION_FOLDER_LIST})
        if (NOT IS_DIRECTORY ${VERSION})
            continue()
        endif ()

        file(GLOB_RECURSE VERSION_SOURCE_FILES 
            ${VERSION}/*.c 
            ${VERSION}/*.h)

        get_filename_component(APPROACH_NAME ${APPROACH} NAME)
        get_filename_component(VERSION_NAME ${VERSION} NAME)

        add_library(nbody_openmp_${APPROACH_NAME}_${VERSION_NAME}_lib STATIC ${COMMON_SOURCE_FILES} ${VERSION_SOURCE_FILES})

        target_include_directories(nbody_openmp_${APPROACH_NAME}_${VERSION_NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(nbody_openmp_${APPROACH_NAME}_${VERSION_NAME}_lib PUBLIC ${DEPENDS_LIBS})

        add_executable(nbody_openmp_${APPROACH_NAME}_${VERSION_NAME} ${MAIN_SOURCE_FILES})
        target_link_libraries(nbody_openmp_${APPROACH_NAME}_${VERSION_NAME} nbody_openmp_${APPROACH_NAME}_${VERSION_NAME}_lib)
        set_target_properties(nbody_openmp_${APPROACH_NAME}_${VERSION_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

        # list(APPEND VALID_IMPLEMENTATION_LIBS_TO_TEST
        #     ${APPROACH_NAME}
        # )

        list(APPEND IMPLEMENTATION_LIBS_TO_TEST
            nbody_openmp_${APPROACH_NAME}_${VERSION_NAME}_lib
        )

        file(READ ${VERSION}/tests TEST_NAMES)

        list (APPEND LIB_TEST
            ${TEST_NAMES}
        )

    endforeach()
endforeach()

# CONFIGURE AND BUILD MPI VERSIONS
find_package(MPI REQUIRED)

file(GLOB MPI_APPROACH_FOLDER_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/*)

foreach(APPROACH ${MPI_APPROACH_FOLDER_LIST})
    if (NOT IS_DIRECTORY ${APPROACH})
        continue()
    endif ()

    file(GLOB MPI_VERSION_FOLDER_LIST ${APPROACH}/*)

    foreach(VERSION ${MPI_VERSION_FOLDER_LIST})
        if (NOT IS_DIRECTORY ${VERSION})
            continue()
        endif ()

        file(GLOB_RECURSE VERSION_SOURCE_FILES 
            ${VERSION}/*.c 
            ${VERSION}/*.h)

        get_filename_component(APPROACH_NAME ${APPROACH} NAME)
        get_filename_component(VERSION_NAME ${VERSION} NAME)

        add_library(nbody_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib STATIC ${COMMON_SOURCE_FILES} ${VERSION_SOURCE_FILES})

        target_include_directories(nbody_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})
        target_link_libraries(nbody_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib PUBLIC ${MPI_C_LIBRARIES} ${DEPENDS_LIBS})

        add_executable(nbody_mpi_${APPROACH_NAME}_${VERSION_NAME} ${MAIN_SOURCE_FILES})
        target_link_libraries(nbody_mpi_${APPROACH_NAME}_${VERSION_NAME} nbody_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib)
        set_target_properties(nbody_mpi_${APPROACH_NAME}_${VERSION_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

        list(APPEND IMPLEMENTATION_LIBS_TO_TEST
            nbody_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib
        )

        # LIST OF TESTS TO RUN FOR EACH IMPLEMENTATION
        file(READ ${VERSION}/tests TEST_NAMES)

        list (APPEND LIB_TEST
            ${TEST_NAMES}
        )

    endforeach()
endforeach()


# CONFIGURE AND BUILD OPENMP + MPI VERSIONS
file(GLOB OPENMP_MPI_APPROACH_FOLDER_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/openmp-mpi/*)

foreach(APPROACH ${OPENMP_MPI_APPROACH_FOLDER_LIST})
    if (NOT IS_DIRECTORY ${APPROACH})
        continue()
    endif ()

    file(GLOB OPENMP_MPI_VERSION_FOLDER_LIST ${APPROACH}/*)

    foreach(VERSION ${OPENMP_MPI_VERSION_FOLDER_LIST})
        if (NOT IS_DIRECTORY ${VERSION})
            continue()
        endif ()

        file(GLOB_RECURSE VERSION_SOURCE_FILES 
            ${VERSION}/*.c 
            ${VERSION}/*.h)

        get_filename_component(APPROACH_NAME ${APPROACH} NAME)
        get_filename_component(VERSION_NAME ${VERSION} NAME)

        add_library(nbody_openmp_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib STATIC ${COMMON_SOURCE_FILES} ${VERSION_SOURCE_FILES})

        target_include_directories(nbody_openmp_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})
        target_link_libraries(nbody_openmp_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib PUBLIC ${MPI_C_LIBRARIES} ${DEPENDS_LIBS})

        add_executable(nbody_openmp_mpi_${APPROACH_NAME}_${VERSION_NAME} ${MAIN_SOURCE_FILES})
        target_link_libraries(nbody_openmp_mpi_${APPROACH_NAME}_${VERSION_NAME} nbody_openmp_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib)
        set_target_properties(nbody_openmp_mpi_${APPROACH_NAME}_${VERSION_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

        list(APPEND IMPLEMENTATION_LIBS_TO_TEST
            nbody_openmp_mpi_${APPROACH_NAME}_${VERSION_NAME}_lib
        )

        # LIST OF TESTS TO RUN FOR EACH IMPLEMENTATION
        file(READ ${VERSION}/tests TEST_NAMES)

        list (APPEND LIB_TEST
            ${TEST_NAMES}
        )

    endforeach()
endforeach()


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

set(ALL_BRUTE_FORCE_TEST
    "COMPUTE_FOCE_TEST-\
    COMPUTE_FOCE_ON_PARTICLE_TEST-\
    COMPUTE_FORCE_IN_NODE_TEST-\
    ALL_MOVE_PARTICLES_TEST-\
    RUN_SIMULATION_TEST"
)

set(ALL_BARNES_HUT_TEST
    "COMPUTE_FOCE_TEST-\
    ALL_MOVE_PARTICLES_TEST-\
    RUN_SIMULATION_TEST"
)

# CONFIGURE BRUTE FORCE TEST TARGET
add_library(nbody_valid_brute_force_lib STATIC ${COMMON_SOURCE_FILES} ${VALID_BRUTE_FORCE_SOURCE_FILES})

target_include_directories(nbody_valid_brute_force_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nbody_valid_brute_force_lib ${DEPENDS_LIBS})

# CONFIGURE BARNES HUT TEST TARGET
add_library(nbody_valid_barnes_hut_lib STATIC ${COMMON_SOURCE_FILES} ${VALID_BARNES_HUT_SOURCE_FILES})

target_include_directories(nbody_valid_barnes_hut_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nbody_valid_barnes_hut_lib ${DEPENDS_LIBS})

foreach(IMPL_LIBS IN ZIP_LISTS IMPLEMENTATION_LIBS_TO_TEST LIB_TEST)

    string(REGEX MATCH "^(.*)brute_force(.*)$" BRUTE_FORCE_APPROACH ${IMPL_LIBS_0})
    string(REGEX MATCH "^(.*)barnes_hut(.*)$" BARNES_HUT_APPROACH ${IMPL_LIBS_0})

    if (NOT ${BRUTE_FORCE_APPROACH} STREQUAL "")
        # CONFIGURE AND BUILD BRUTE FORCE TESTS
        add_executable(${IMPL_LIBS_0}_test ${BRUTE_FORCE_TESTS_SOURCE_FILES})
        target_link_libraries(${IMPL_LIBS_0}_test ${IMPL_LIBS_0} nbody_valid_brute_force_lib ${TEST_DEPENDS_LIBS})
        set_target_properties(${IMPL_LIBS_0}_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/tests)
        
        string(REGEX MATCH "^(.*)mpi(.*)$" MPI_APPROACH ${IMPL_LIBS_0})

        # CONFIGURE MPI TESTS MANUALLY TO LAUNCH IT VIA MPIRUN
        if (NOT ${MPI_APPROACH} STREQUAL "")
            add_test(
                NAME ${IMPL_LIBS_0}_test
                COMMAND mpirun.mpich -np 5 ${CMAKE_CURRENT_BINARY_DIR}/bin/tests/${IMPL_LIBS_0}_test ${TEST_PARAMETERS})

            set_property(TEST ${IMPL_LIBS_0}_test PROPERTY ENVIRONMENT "RDMAV_FORK_SAFE=1") 
        else()
            add_test(NAME ${IMPL_LIBS_0}_test COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/tests/${IMPL_LIBS_0}_test ${TEST_PARAMETERS})
        endif()
    
    elseif (NOT ${BARNES_HUT_APPROACH} STREQUAL "")
        # CONFIGURE AND BUILD BRUTE FORCE TESTS
        add_executable(${IMPL_LIBS_0}_test ${BARNES_HUT_TESTS_SOURCE_FILES})
        target_link_libraries(${IMPL_LIBS_0}_test ${IMPL_LIBS_0} nbody_valid_barnes_hut_lib ${TEST_DEPENDS_LIBS})
        set_target_properties(${IMPL_LIBS_0}_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/tests)

        string(REGEX MATCH "^(.*)mpi(.*)$" MPI_APPROACH ${IMPL_LIBS_0})

        # CONFIGURE MPI TESTS MANUALLY TO LAUNCH IT VIA MPIRUN
        if (NOT ${MPI_APPROACH} STREQUAL "")
            add_test(
                NAME ${IMPL_LIBS_0}_test
                COMMAND mpirun.mpich -np 5 ${CMAKE_CURRENT_BINARY_DIR}/bin/tests/${IMPL_LIBS_0}_test ${TEST_PARAMETERS})

            set_property(TEST ${IMPL_LIBS_0}_test PROPERTY ENVIRONMENT "RDMAV_FORK_SAFE=1") 
        else()
            add_test(NAME ${IMPL_LIBS_0}_test COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/tests/${IMPL_LIBS_0}_test ${TEST_PARAMETERS})
        endif()

    endif ()

    string(REPLACE "-" ";" TEST_LIST ${IMPL_LIBS_1})

    foreach(TEST ${TEST_LIST})
        # REMOVE WHITESPACES
        string(STRIP ${TEST} TEST_NAME)
        target_compile_options(${IMPL_LIBS_0}_test PRIVATE -D${TEST_NAME})
    endforeach()
endforeach()