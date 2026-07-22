cmake_minimum_required(VERSION 3.24)
project(mochi_sandbox_windows VERSION 0.1.0 LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "mochi-sandbox-windows can only be built with the Windows SDK")
endif()

add_executable(mochi-sandbox-windows src/main.cpp)
target_compile_features(mochi-sandbox-windows PRIVATE cxx_std_17)
target_compile_definitions(mochi-sandbox-windows PRIVATE UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)

if(MSVC)
  target_compile_options(mochi-sandbox-windows PRIVATE /W4 /WX /permissive- /guard:cf)
  target_link_options(mochi-sandbox-windows PRIVATE /DYNAMICBASE /NXCOMPAT /GUARD:CF)
endif()

set(
  MOCHI_SANDBOX_OUTPUT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../mochi/runtime/sandbox/bin"
  CACHE PATH
  "Directory for the packaged Windows sandbox helper"
)

set_target_properties(
  mochi-sandbox-windows
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${MOCHI_SANDBOX_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${MOCHI_SANDBOX_OUTPUT_DIR}"
)

install(TARGETS mochi-sandbox-windows RUNTIME DESTINATION mochi/runtime/sandbox/bin)
install(FILES protocol.schema.json DESTINATION share/mochi-sandbox-windows)
