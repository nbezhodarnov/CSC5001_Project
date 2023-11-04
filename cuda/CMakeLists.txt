cmake_minimum_required(VERSION 3.24)
# set the project name
project(nbody_cuda_brute_force LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)


# CONFIGURE AND BUILD OPENMP VERSIONS
file(GLOB CUDA_APPROACH_FOLDER_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(APPROACH ${CUDA_APPROACH_FOLDER_LIST})
    if (NOT IS_DIRECTORY ${APPROACH})
        continue()
    endif ()

    file(GLOB VERSION_FOLDER_LIST ${APPROACH}/*)

    foreach(VERSION ${VERSION_FOLDER_LIST})
        if (NOT IS_DIRECTORY ${VERSION})
            continue()
        endif ()

        file(GLOB_RECURSE VERSION_SOURCE_FILES 
            ${VERSION}/*.cu)

        get_filename_component(APPROACH_NAME ${APPROACH} NAME)
        get_filename_component(VERSION_NAME ${VERSION} NAME)

        add_library(nbody_cuda_${APPROACH_NAME}_${VERSION_NAME}_lib STATIC ${COMMON_SOURCE_FILES} ${VERSION_SOURCE_FILES})
        set_target_properties(nbody_cuda_${APPROACH_NAME}_${VERSION_NAME}_lib PROPERTIES CUDA_ARCHITECTURES native)

        target_include_directories(nbody_cuda_${APPROACH_NAME}_${VERSION_NAME}_lib PUBLIC ${CMAKE_SOURCE_DIR})
        target_link_libraries(nbody_cuda_${APPROACH_NAME}_${VERSION_NAME}_lib PUBLIC ${DEPENDS_LIBS} ${CUDA_LIBRARIES})

        add_executable(nbody_cuda_${APPROACH_NAME}_${VERSION_NAME} ${MAIN_SOURCE_FILES})
        target_link_libraries(nbody_cuda_${APPROACH_NAME}_${VERSION_NAME} nbody_cuda_${APPROACH_NAME}_${VERSION_NAME}_lib)
        set_target_properties(nbody_cuda_${APPROACH_NAME}_${VERSION_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

        # CONGIURE TESTS
        set(IMPLEMENTATION_LIBS_TO_TEST ${IMPLEMENTATION_LIBS_TO_TEST} nbody_cuda_${APPROACH_NAME}_${VERSION_NAME}_lib PARENT_SCOPE)
        set(LIB_TEST ${LIB_TEST} "RUN_SIMULATION_TEST" PARENT_SCOPE)

    endforeach()
endforeach()