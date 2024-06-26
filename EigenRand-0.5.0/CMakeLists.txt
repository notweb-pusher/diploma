cmake_minimum_required(VERSION 3.9)

project(EigenRand VERSION 0.5.0 DESCRIPTION "EigenRand, Random Number Generator for Eigen")

option(EIGENRAND_BUILD_TEST "Build Test sets" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE true)
set(ADDITIONAL_FLAGS "${EIGENRAND_CXX_FLAGS}")

if (MSVC)
    set(CMAKE_C_FLAGS_DEBUG "-DDEBUG -DC_FLAGS -Zi -Od ${ADDITIONAL_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")

    set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DRELEASE -DC_FLAGS -O2 -Oi -Gy ${ADDITIONAL_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -Zi")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
else ()
    link_libraries(pthread)

    set(CMAKE_C_FLAGS_DEBUG "-DDEBUG -DC_FLAGS -g3 -O0 ${ADDITIONAL_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-DDEBUG -DLINKER_FLAGS")

    set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DRELEASE -DC_FLAGS -O3 ${ADDITIONAL_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-DRELEASE -DLINKER_FLAGS")

    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -g3")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
endif ()

include_directories(.)

add_executable("${PROJECT_NAME}-benchmark"
        benchmark/benchmark.cpp
)

add_executable("${PROJECT_NAME}-benchmark-mv"
        benchmark/benchmark_mv.cpp
)

add_executable("${PROJECT_NAME}-benchmark-disc"
        benchmark/benchmark_disc.cpp
)

add_executable("${PROJECT_NAME}-benchmark-vop"
        benchmark/benchmark_vectorize_over_params.cpp
)

add_executable("${PROJECT_NAME}-accuracy"
        benchmark/accuracy.cpp
)

if (EIGENRAND_BUILD_TEST)
    add_subdirectory(googletest)
    add_subdirectory(test)
endif ()
