cmake_minimum_required(VERSION 4.0)
#set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
#set(VCPKG_TARGET_TRIPLET "wasm32-emscripten" CACHE STRING "wasm triplet" FORCE)
set(DAWN_FETCH_DEPENDENCIES ON)

project(OptimalPixelTransport)
set(CMAKE_CXX_STANDARD 20)

set(EM_PROFILING_FLAGS
        --profiling-funcs
        -g3
)

#file(GLOB_RECURSE CORE_SRC src/.cpp)

add_executable(${PROJECT_NAME}
        #        ${CORE_SRC}
        src/main.cpp
        src/particle.cpp
        src/particle_buffer.cpp
        src/image_stb.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        include/
        Shaders/
)

add_subdirectory("dawn" EXCLUDE_FROM_ALL)
add_subdirectory("eigen-5.0.0")
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

if (EMSCRIPTEN)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".html")
    target_link_libraries(${PROJECT_NAME} PRIVATE emdawnwebgpu_cpp webgpu_glfw)
    target_link_options(${PROJECT_NAME} PRIVATE
            "-sASYNCIFY=1"
            "-sUSE_GLFW=3"
            "-sSTB_IMAGE=1"
            "-sALLOW_MEMORY_GROWTH=1"
            "-sNO_DISABLE_EXCEPTION_CATCHING"
            "-sSTACK_SIZE=10048576"
            #            "-sSTACK_OVERFLOW_CHECK=2"
            "-sASSERTIONS=1"
            "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/Shaders@/Shaders/"
            "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/Images/img_1.png@/img_1.png"
            "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/Images/img_6.png@/img_6.png"
    )
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
    add_compile_options(${EM_PROFILING_FLAGS})
    add_link_options(${EM_PROFILING_FLAGS})
else ()
    target_link_libraries(${PROJECT_NAME} PRIVATE webgpu_dawn webgpu_glfw glfw)
endif ()