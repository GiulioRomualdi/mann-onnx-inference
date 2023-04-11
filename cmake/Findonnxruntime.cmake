# SPDX-FileCopyrightText: 2023 Giulio Romualdi
# SPDX-License-Identifier: BSD-3-Clause

#[=======================================================================[.rst:
Findonnxruntime
-----------

The following imported targets are created:

onnxruntime::onnxruntime

#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_path(onnxruntime_INCLUDE_DIR
  NAMES onnxruntime_c_api.h
  HINTS "${onnxruntime_SOURCE_DIR}"
  ENV onnxruntime_SOURCE_DIR
  PATH_SUFFIXES include)
mark_as_advanced(onnxruntime_INCLUDE_DIR)
find_library(onnxruntime_LIBRARY
  NAMES libonnxruntime.so
  PATH_SUFFIXES lib libs)

mark_as_advanced(onnxruntime_LIBRARY)

find_package_handle_standard_args(onnxruntime DEFAULT_MSG onnxruntime_INCLUDE_DIR onnxruntime_LIBRARY)

if(onnxruntime_FOUND)
  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}")
    set_property(TARGET onnxruntime::onnxruntime PROPERTY
      IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
  endif()
endif()
