cmake_minimum_required(VERSION 3.24.0)

project(CSC5001-N-BODY VERSION 1.0.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-Wall -fopenmp")

set(DEPENDS_LIBS
    m
    pthread
    check
    subunit
)

set(DUMP_RESULT FALSE CACHE BOOL "Dump the results of the simulation")
set(DISPLAY TRUE CACHE BOOL "Display the results of the simulation")

# COMMON SOURCES
file(GLOB_RECURSE COMMON_SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/nbody/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/nbody/*.h)

# UI SOURCES
file(GLOB_RECURSE UI_SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/ui/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/ui/*.h)

# BRUTE FORCE SOURCES
set(NBODY_BRUTE_FORCE_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/sequential/nbody_brute_force.c
        ${CMAKE_CURRENT_SOURCE_DIR}/sequential/nbody_brute_force.h
    )

# BARNES HUT SOURCES
set(NBODY_BARNES_HUT_SOURCE_FILES
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


# CONFIGURE AND BUILD BRUTE FORCE VERSION
add_library(nbody_brute_force_lib STATIC ${COMMON_SOURCE_FILES} ${NBODY_BRUTE_FORCE_SOURCE_FILES})

target_include_directories(nbody_brute_force_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nbody_brute_force_lib ${DEPENDS_LIBS})

add_executable(nbody_brute_force ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_brute_force nbody_brute_force_lib)
set_target_properties(nbody_brute_force PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# CONFIGURE AND BUILD BARNES HUT VERSION
add_library(nbody_barnes_hut_lib STATIC ${COMMON_SOURCE_FILES} ${NBODY_BARNES_HUT_SOURCE_FILES})

target_include_directories(nbody_barnes_hut_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nbody_barnes_hut_lib ${DEPENDS_LIBS})

add_executable(nbody_barnes_hut ${MAIN_SOURCE_FILES})
target_link_libraries(nbody_barnes_hut nbody_barnes_hut_lib)
set_target_properties(nbody_barnes_hut PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
