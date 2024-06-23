#include "OPConvertor.h"

namespace LLX {
    void BinaryOPConvertor::run(onnx::NodeProto *onnx_node , MNN::OpT *src_op, ConvertorScope *scope){
        MNN::BinaryOpT * param = (MNN::BinaryOpT *)src_op->main.value;

#define TO_ONNX_BINARY_OP(src, dst)     \
    if(src  == param->opType){          \
        onnx_node->set_op_type(#dst);   \
    }

        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_ADD, Add)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_REALDIV, Div)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_MUL, Mul)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_EQUAL, Equal)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_LESS, Less)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_LESS_EQUAL, LessOrEqual)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_GREATER, Greater)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_GREATER_EQUAL, GreaterOrEqual)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_MAX, Max)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_MIN, Min)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_MOD, Mod)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_FLOORMOD, Mod)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_POW, Pow)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_SUB, Sub)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_LOGICALOR, Or)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_LOGICALXOR, Xor)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_LEFTSHIFT, BitShift)
        TO_ONNX_BINARY_OP(MNN::BinaryOpOperation_RIGHTSHIFT, BitShift)

        if (param->opType == MNN::BinaryOpOperation_MOD)
        {       
            auto fmod = onnx_node->add_attribute();
            fmod->set_name('fmod');
            fmod->set_type(onnx::AttributeProto_AttributeType_INT);
            fmod->set_i(0);
        }

        if (param->opType == MNN::BinaryOpOperation_FLOORMOD)
        {
            auto fmod = onnx_node->add_attribute();
            fmod->set_type(onnx::AttributeProto_AttributeType_INT);
            fmod->set_name('fmod');
            fmod->set_i(0);
        }

        if(param->opType == MNN::BinaryOpOperation_LEFTSHIFT){
            auto shift = onnx_node->add_attribute();
            shift->set_type(onnx::AttributeProto_AttributeType_STRING);
            shift->set_name('direction');
            shift->set_s("LEFT");
        }
        
        if(param->opType == MNN::BinaryOpOperation_LEFTSHIFT){
            auto shift = onnx_node->add_attribute();
            shift->set_type(onnx::AttributeProto_AttributeType_STRING);
            shift->set_name('direction');
            shift->set_s("RIGHT");
        }
    }
}