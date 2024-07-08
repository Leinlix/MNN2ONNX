#include "OPConvertor.h"

namespace LLX{
    onnx::TensorProto_DataType OPConvertor::convertDataType(const MNN::DataType type){
        static std::map<MNN::DataType, onnx::TensorProto_DataType> data_type_map = {
            {MNN::DataType_DT_FLOAT, onnx::TensorProto_DataType_FLOAT},
            {MNN::DataType_DT_HALF, onnx::TensorProto_DataType_FLOAT16},
            {MNN::DataType_DT_BFLOAT16, onnx::TensorProto_DataType_BFLOAT16},
            {MNN::DataType_DT_INT8, onnx::TensorProto_DataType_INT8},
            {MNN::DataType_DT_INT16, onnx::TensorProto_DataType_INT16},
            {MNN::DataType_DT_INT32, onnx::TensorProto_DataType_INT32},
            {MNN::DataType_DT_INT64, onnx::TensorProto_DataType_INT64},
            {MNN::DataType_DT_BOOL, onnx::TensorProto_DataType_BOOL},
            {MNN::DataType_DT_DOUBLE, onnx::TensorProto_DataType_DOUBLE},
            {MNN::DataType_DT_UINT8, onnx::TensorProto_DataType_UINT8},
            {MNN::DataType_DT_UINT16, onnx::TensorProto_DataType_UINT16}
        };
        if(data_type_map.find(type)==data_type_map.end()){
            std::cout<< "Unsupported data type convert"<< type <<std::endl;
            return onnx::TensorProto_DataType_UNDEFINED;
        }
        return data_type_map[type];
    }

    onnx::TensorProto * OPConvertor::convertBlobToTensor(const MNN::BlobT *blob){
        auto tensor = new onnx::TensorProto();
        auto data_type = convertDataType(blob->dataType);
        tensor->set_data_type(data_type);
        auto dim_size = blob->dims.size();
        size_t data_size = 1;
        for(int i = 0; i < dim_size ; i ++){
            tensor->add_dims(blob->dims[i]);
            data_size *= blob->dims[i];
        }
        switch (blob->dataType) 
        {
        case MNN::DataType_DT_INT8:
            for(int j = 0 ; j < data_size ; j++){
                tensor->add_int32_data(blob->int8s[j]);
            }
            break;
         case MNN::DataType_DT_UINT8:
            for(int j = 0 ; j < data_size ; j++){
                tensor->add_int32_data(blob->uint8s[j]);
            }
            break;
         case MNN::DataType_DT_INT32:
            for(int j = 0 ; j < data_size ; j++){
                tensor->add_int32_data(blob->int32s[j]);
            }
            break;
         case MNN::DataType_DT_INT64:
            for(int j = 0 ; j < data_size ; j++){
                tensor->add_int32_data(blob->int64s[j]);
            }
            break;
         case MNN::DataType_DT_FLOAT:
            for(int j = 0 ; j < data_size ; j++){
                tensor->add_int32_data(blob->float32s[j]);
            }
            break;
        
        default:
            std::cout<< "unsupported blob data type: "<< blob->dataType<<std::endl;
            break;
        }
        return tensor;
    }

#pragma mark - OPConvertorManager

    OPConvertorManager * OPConvertorManager::manager = nullptr;
    OPConvertorManager * OPConvertorManager::sharedInstance(){
        if (manager == nullptr)
        {
            manager = new OPConvertorManager();
        }
        return manager;
    }

    onnx::ValueInfoProto * OPConvertor::valueInfoProtoFromArgs(const std::string &name, int type, int dim_size , std::vector<int> dims ){
        onnx::ValueInfoProto * value = new onnx::ValueInfoProto();
        value->set_name(name);
        value->mutable_type()->mutable_tensor_type()->set_elem_type(type);
        auto shape = value->mutable_type()->mutable_tensor_type()->mutable_shape();
        for(int i = 0 ; i < dim_size; i++){
            shape->add_dim()->set_dim_value(dims[i]);
        }
        return value;
    }

    onnx::ValueInfoProto * OPConvertor::valueInfoProtoFromInitialzer(onnx::TensorProto * tensor){
        std::vector<int> dims;
        for(int i = 0 ; i < tensor->dims().size(); i++){
            dims.push_back(tensor->dims(i));
        }
        auto  value = valueInfoProtoFromArgs((std::string)tensor->name() ,tensor->data_type(), tensor->dims().size(),dims);
        return value;
    }

    onnx::TensorProto * OPConvertor::tensorProtoFromOp(MNN::OpT *op){
        auto op_param = op->main.value;
        onnx::TensorProto *init = LLX::OPConvertor::convertBlobToTensor((MNN::BlobT *)op_param);
        init->set_name(op->name);
        return init;
    }

    OPConvertorManager::OPConvertorManager(){
        // add register
        REGISTER_OP_CONVERTOR(MNN::OpType_ArgMax ,ArgMaxOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_ArgMin, ArgMinOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_BinaryOp, BinaryOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_Cast, CastOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_ConvertTensor, ConvertTensorOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_ConvolutionDepthwise, ConvolutionDepthwiseOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_Convolution, ConvolutionOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_Interp, InterpOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_Pooling, PoolingOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_Pooling3D, PoolingOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_ReLU, ReluOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_Selu, SeluOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_Sigmoid, SigmoidOPConvertor)
        REGISTER_OP_CONVERTOR(MNN::OpType_UnaryOp, UnaryOPConvertor)
    }

    void OPConvertorManager::registerNodeConvertor(MNN::OpType op_type , OPConvertor * op_convertor){
        if(convertor_map.find(op_type)!=convertor_map.end()){
            std::cout<< "already existed op convertor ! please check OPConvertor register."<<std::endl;
        }
        std::pair<MNN::OpType , OPConvertor *> entity = std::make_pair(op_type, op_convertor);
        convertor_map.insert(entity);
    }

    void OPConvertorManager::convertToNode(MNN::OpT *op, onnx::NodeProto *node , ConvertorScope *scope){
        MNN::OpType type = op->type;
        auto it = convertor_map.find(type);
        if(it != convertor_map.end()){
            OPConvertor *convertor = it->second;
            convertor->run(node, op, scope);
        }else{
            std::cout<< "error in node convert, unsupported op : "<<op->name<<std::endl;
        }
    }

    void OPConvertorManager::convertAfter(MNN::OpT *op, onnx::NodeProto *node , ConvertorScope *scope){
        if(MNN::OpType_Convolution == op->type || MNN::OpType_ConvolutionDepthwise == op->type){
            MNN::Convolution2DT *param = (MNN::Convolution2DT *)op->main.value;
            if(param->common->relu){
                // find ouput value 
                auto node_output_name = node->input(0);
                auto val =  scope->findValueInfoByName(node_output_name);
                // build relu node 
                auto relu = new onnx::NodeProto();
                auto relu_n = op->name + "/relu";
                relu->set_name(relu_n);
                relu->set_op_type("Relu");
                // build relu input 
                auto relu_i_n = relu_n + "_input";
                std::vector<int> dims;
                int dim_size = val->type().tensor_type().shape().dim_size();
                for(int i = 0 ; i < dim_size; i++){
                    dims.push_back(val->type().tensor_type().shape().dim(i).dim_value());
                }
                auto relu_input = OPConvertor::valueInfoProtoFromArgs(relu_i_n, onnx::TensorProto_DataType_FLOAT,dim_size, dims);
                scope->pushValue(relu_input);
                scope->pushNode(relu);
                // change output
                node->clear_output();
                node->add_output(relu_i_n);
                relu->add_input(relu_i_n);
                relu->add_output(node_output_name);
            }else if (param->common->relu6){
                // find ouput value 
                auto node_output_name = node->input(0);
                auto val =  scope->findValueInfoByName(node_output_name);
                // insert clip node 
                auto clip = new onnx::NodeProto();
                auto clip_n = op->name + "/clip";
                clip->set_name(clip_n);
                clip->set_op_type("Clip");
                // add input 
                auto name_min = clip_n + "_min";
                auto name_max = clip_n + "_max";
                std::vector<int> dims_max, dims_min;
                dims_max.push_back(1);
                dims_min.push_back(1);
                auto clip_min = OPConvertor::valueInfoProtoFromArgs(name_min, onnx::TensorProto_DataType_FLOAT, 1, dims_min);
                auto clip_max = OPConvertor::valueInfoProtoFromArgs(name_max, onnx::TensorProto_DataType_FLOAT, 1, dims_max);
                // initializer 
                auto min_init = new onnx::TensorProto();
                min_init->set_name(name_min);
                min_init->set_data_type(onnx::TensorProto_DataType_FLOAT);
                min_init->add_float_data(0.0);
                auto max_init = new onnx::TensorProto();
                max_init->set_name(name_max);
                max_init->set_data_type(onnx::TensorProto_DataType_FLOAT);
                max_init->add_float_data(6.0);
                // value info 
                auto clip_i_n = clip_n + "_input";
                std::vector<int> dims;
                int dim_size = val->type().tensor_type().shape().dim_size();
                for(int i = 0 ; i < dim_size; i++){
                    dims.push_back(val->type().tensor_type().shape().dim(i).dim_value());
                }
                auto clip_input = OPConvertor::valueInfoProtoFromArgs(clip_i_n,onnx::TensorProto_DataType_FLOAT, dim_size, dims);
                scope->pushInitializer(min_init);
                scope->pushInitializer(max_init);
                scope->pushValue(clip_input);
                scope->pushNode(clip);
                // change output
                node->clear_output();
                node->add_output(clip_i_n);
                clip->add_input(clip_i_n);
                clip->add_input(name_min);
                clip->add_input(name_max);
                clip->add_output(node_output_name);
            }
        }else if (MNN::OpType_Interp == op->type){
            // change the order of inputs
            auto a = node->input(0);
            auto b = node->input(1);
            auto c = node->input(2);
            auto d = node->input(3);
            node->clear_input();
            node->add_input(a);
            node->add_input(c);
            node->add_input(d);
            node->add_input(b);
        }
    }
}