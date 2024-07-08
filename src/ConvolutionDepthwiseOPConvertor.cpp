#include "OPConvertor.h"

namespace LLX{
    void ConvolutionDepthwiseOPConvertor::run(onnx::NodeProto *onnx_node , MNN::OpT *src_op, ConvertorScope *scope){
        MNN::Convolution2DT * param = (MNN::Convolution2DT *)src_op->main.value;
        onnx_node->set_op_type("Conv");

        auto kernel_shape = onnx_node->add_attribute();
        kernel_shape->set_name("kernel_shape");
        kernel_shape->set_type(onnx::AttributeProto_AttributeType_INTS);
        kernel_shape->add_ints(param->common->kernelX);
        kernel_shape->add_ints(param->common->kernelY);

        auto dilations = onnx_node->add_attribute();
        dilations->set_name("dilations");
        dilations->set_type(onnx::AttributeProto_AttributeType_INTS);
        dilations->add_ints(param->common->dilateX);
        dilations->add_ints(param->common->dilateY);

        auto pads = onnx_node->add_attribute();
        pads->set_name("pads");
        pads->set_type(onnx::AttributeProto_AttributeType_INTS);
        pads->add_ints(param->common->padX);
        pads->add_ints(param->common->padY);

        auto strides = onnx_node->add_attribute();
        strides->set_name("strides");
        strides->set_type(onnx::AttributeProto_AttributeType_INTS);
        strides->add_ints(param->common->strideX);
        strides->add_ints(param->common->strideY);

        auto group = onnx_node->add_attribute();
        group->set_name("group");
        group->set_type(onnx::AttributeProto_AttributeType_INT);
        group->set_i(param->common->group);

        // change params to input 
        if(!param->weight.empty()){
            std::string name_w = src_op->name + "/weight";
            onnx::TensorProto *weight = new onnx::TensorProto();
            weight->set_name(name_w);
            weight->add_dims(param->common->outputCount);
            // depthwise will change the second dimension
            weight->add_dims(1);
            weight->add_dims(param->common->kernelX);
            weight->add_dims(param->common->kernelY);
            weight->set_data_type(onnx::TensorProto_DataType_FLOAT);
            for(int j = 0; j < param->weight.size(); j++){
                weight->add_float_data(param->weight[j]);
            }

            onnx::ValueInfoProto *weight_i = new onnx::ValueInfoProto();
            weight_i->set_name(name_w);
            auto tpp = weight_i->mutable_type()->mutable_tensor_type();
            tpp->set_elem_type(onnx::TensorProto_DataType_FLOAT);
            // output * input * kx * ky
            onnx::TensorShapeProto_Dimension *dim_o = new onnx::TensorShapeProto_Dimension();
            dim_o->set_dim_value(param->common->outputCount);
            tpp->mutable_shape()->mutable_dim()->AddAllocated(dim_o);

            onnx::TensorShapeProto_Dimension *dim_i = new onnx::TensorShapeProto_Dimension();
            dim_i->set_dim_value(param->common->inputCount);
            tpp->mutable_shape()->mutable_dim()->AddAllocated(dim_i);

            onnx::TensorShapeProto_Dimension *dim_x = new onnx::TensorShapeProto_Dimension();
            dim_x->set_dim_value(param->common->kernelX);
            tpp->mutable_shape()->mutable_dim()->AddAllocated(dim_x);
            
            onnx::TensorShapeProto_Dimension *dim_y = new onnx::TensorShapeProto_Dimension();
            dim_y->set_dim_value(param->common->kernelY);
            tpp->mutable_shape()->mutable_dim()->AddAllocated(dim_y);

            onnx_node->add_input(name_w);
            scope->pushInitializer(weight);
            scope->pushValue(weight_i);
        }

        if(!param->bias.empty()){
            std::string name_b = src_op->name+"/bias";

            onnx::TensorProto *bias = new onnx::TensorProto();
            bias->set_name(name_b);
            bias->add_dims(param->common->outputCount);
            bias->set_data_type(onnx::TensorProto_DataType_FLOAT);
            for(int j = 0; j < param->bias.size(); j++){
                bias->add_float_data(param->bias[j]);
            }

            onnx::ValueInfoProto *bias_i = new onnx::ValueInfoProto();
            bias_i->set_name(name_b);
            bias_i->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_FLOAT);
            onnx::TypeProto_Tensor *tpp = bias_i->mutable_type()->mutable_tensor_type();
            onnx::TensorShapeProto_Dimension *dim = new onnx::TensorShapeProto_Dimension();
            dim->set_dim_value(param->common->outputCount);
            tpp->mutable_shape()->mutable_dim()->AddAllocated(dim);

            onnx_node->add_input(name_b);
            scope->pushInitializer(bias);
            scope->pushValue(bias_i);
       }
    }
}