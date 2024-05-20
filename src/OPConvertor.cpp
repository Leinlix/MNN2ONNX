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

    OPConvertorManager::OPConvertorManager(){
        // add register
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
        // wait to add 
    }
}