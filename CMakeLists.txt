cmake_minimum_required(VERSION 3.11)

project(mnn2onnx)

SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)
SET(CMAKE_BUILD_TYPE "Debug")

#flatbuffers
add_subdirectory(3d_party/flatbuffers
    ${CMAke_CURRENT_BINARY_DIR}/flatbuffers-build
    EXCLUDE_FROM_ALL)
include_directories(3d_party/flatbuffers/include)

#protobuffer
add_subdirectory(3d_party/protobuf/cmake)
add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)
SET(Protobuf_LIBRARIES libprotobuf)
include_directories(3d_party/protobuf/src)

#mnn header
add_library(MNN_Gen INTERFACE 3d_party/mnn/CaffeOp_generated.h
    3d_party/mnn/ExtraInfo_generated.h
    3d_party/mnn/MNN_generated.h
    3d_party/mnn/Tensor_generated.h
    3d_party/mnn/TensorflowOp_generated.h
    3d_party/mnn/TFQuantizeOp_generated.h
    3d_party/mnn/TrainInfo_generated.h
    3d_party/mnn/Type_generated.h
    3d_party/mnn/UserDefine_generated.h)
include_directories(3d_party/mnn)

#onnx header
add_library(ONNX_Gen STATIC 3d_party/onnx/onnx.pb.cc)
target_include_directories(ONNX_Gen PUBLIC 3d_party/onnx)

#source 
include_directories(include)
add_executable(mnn2onnx src/Options.cpp)

target_link_libraries(mnn2onnx PUBLIC ${Protobuf_LIBRARIES})

target_link_libraries(mnn2onnx PRIVATE flatbuffers)

target_link_libraries(mnn2onnx PRIVATE MNN_Gen)

target_link_libraries(mnn2onnx PRIVATE ONNX_Gen)