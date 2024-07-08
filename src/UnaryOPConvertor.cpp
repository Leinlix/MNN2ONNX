#include "OPConvertor.h"

namespace LLX{
    void UnaryOPConvertor::run(onnx::NodeProto *onnx_node , MNN::OpT *src_op, ConvertorScope *scope){
        MNN::UnaryOpT * unary_param = (MNN::UnaryOpT*)src_op->main.value;
        auto attr_proto = onnx_node->add_attribute();
        attr_proto->set_type(onnx::AttributeProto_AttributeType_FLOAT);
#define TO_ONNX_OP_TYPE(src, dst)       \
    if(unary_param->opType == src){onnx_node->set_op_type(dst);}
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ABS,"Abs")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ACOS,"Acos")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ACOSH,"Acosh")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ASIN,"Asin")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ASINH,"Asinh")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ATAN,"Atan")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ATANH,"Atanh")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_CEIL,"Ceil")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_COS,"Cos")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_COSH,"Cosh")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_EXP,"Exp")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ERF,"Erf")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ERFC,"Erfc")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ERFINV,"Erfinv")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_EXPM1,"Expm1")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_FLOOR,"Floor")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_HARDSWISH,"HardSwish")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_LOG,"Log")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_LOG1P,"Log1p")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_GELU,"Gelu")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_NEG,"Neg")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_SIN,"Sin")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_SINH,"Sinh")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_SQRT,"Sqrt")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_TAN,"Tan")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_TANH,"Tanh")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_RECIPROCAL,"Reciprocal")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_ROUND,"Round")
        else
        TO_ONNX_OP_TYPE(MNN::UnaryOpOperation_SIGN,"Sign")
        else{
            std::cout<<"unsupported unary op : "<<unary_param->opType<<std::endl;
        }
        
        
    }
}