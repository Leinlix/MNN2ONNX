#include "OPConvertor.h"

namespace LLX{
    void ReluOPConvertor::run(onnx::NodeProto *onnx_node , MNN::OpT *src_op, ConvertorScope *scope){
        MNN::ReluT * p = (MNN::ReluT*)src_op->main.value;
        if(p->slope!=.0f){
            onnx_node->set_op_type("LeakyRelu");
            auto alpha = onnx_node->add_attribute();
            alpha->set_name("alpha");
            alpha->set_type(onnx::AttributeProto_AttributeType_FLOAT);
            alpha->set_f(p->slope);
        }else{
            onnx_node->set_op_type("Relu");
        }
    }
}