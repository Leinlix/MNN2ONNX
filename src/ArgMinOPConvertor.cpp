#include "OPConvertor.h"

namespace LLX {
    void ArgMinOPConvertor::run(onnx::NodeProto *onnx_node , MNN::OpT *src_op, ConvertorScope *scope){
        onnx_node->set_op_type("ArgMin");
        MNN::ArgMaxT *param = (MNN::ArgMaxT *) src_op->main.value;

        auto axis = onnx_node->add_attribute();
        axis->set_name("axis");
        axis->set_type(onnx::AttributeProto_AttributeType_INT);
        axis->set_i(param->axis);

        // keepdims = 1 will split to argmin and squeeze
        auto keepdims = onnx_node->add_attribute();
        keepdims->set_name("keepdims");
        keepdims->set_type(onnx::AttributeProto_AttributeType_INT);
        keepdims->set_i(0);

        // mnn ignore the select_last_index
    }
}