#include "OPConvertor.h"

namespace LLX{
    void SigmoidOPConvertor::run(onnx::NodeProto *onnx_node , MNN::OpT *src_op, ConvertorScope *scope){
        onnx_node->set_op_type("Sigmoid");
    }
}