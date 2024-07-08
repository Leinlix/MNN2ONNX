#include "OPConvertor.h"

namespace LLX{
    void SeluOPConvertor::run(onnx::NodeProto *onnx_node , MNN::OpT *src_op, ConvertorScope *scope){
        MNN::SeluT * param = (MNN::SeluT *)src_op->main.value;
        onnx_node->set_op_type("Selu");

        auto alpha = onnx_node->add_attribute();
        alpha->set_name("alpha");
        alpha->set_type(onnx::AttributeProto_AttributeType_FLOAT);
        alpha->set_f(param->alpha);

        auto gamma = onnx_node->add_attribute();
        gamma->set_name("gamma");
        gamma->set_type(onnx::AttributeProto_AttributeType_FLOAT);
        gamma->set_f(param->scale);
    }
}