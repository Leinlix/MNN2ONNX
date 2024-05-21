#include "Convertor.h"
#include "OPConvertor.h"

#include <iostream>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <fstream>

namespace LLX{

    void Convertor::loadModel(std::string &input_path){
        std::ifstream input_file(input_path.c_str(), std::ios::binary);
        input_file.seekg(0, std::ios::end);
        const auto size  = input_file.tellg();
        input_file.seekg(0, std::ios::beg);
        if(size == -1){
            std::cout<<"Invalid model input path! "<< input_path <<std::endl;
        }else{
            char *buffer = new char[size];
            input_file.read(buffer,size);
            input_file.close();
            auto net = MNN::UnPackNet(buffer);
            mnn_net_ptr = std::move(net);
            delete[] buffer;
        }
    }

    void Convertor::convertModel(){
        onnx_model = new onnx::ModelProto();
        scope = new ConvertorScope(mnn_net_ptr.get(), onnx_model->mutable_graph());
        for(int i = 0 ; i < mnn_net_ptr->oplists.size(); i ++){
            std::vector<std::string> outputNames = mnn_net_ptr->outputName;
            auto op = mnn_net_ptr->oplists[i].get();
            if(op->type == MNN::OpType_Const || op->type == MNN::OpType_TrainableParam){
                auto initializer = OPConvertor::tensorProtoFromOp(op);
                auto value = OPConvertor::valueInfoProtoFromInitialzer(initializer);
                scope->pushInitializer(initializer);
                scope->pushValue(value);
            }else if(op->type == MNN::OpType_Input){
                MNN::InputT *input_param = (MNN::InputT *)op->main.value;
                onnx::TensorProto_DataType dtype = OPConvertor::convertDataType(input_param->dtype);
                std::vector<int> input_dims ;
                for(int j = 0 ; j < input_param->dims.size(); j++){
                    input_dims.push_back(input_param->dims[j]);
                }
                onnx::ValueInfoProto  *value = OPConvertor::valueInfoProtoFromArgs(op->name, dtype, input_param->dims.size(), input_dims);
                scope->pushInput(value);
            }else{
                onnx::NodeProto *node = new onnx::NodeProto();
                node->set_name(op->name);
                // input 
                int input_idx_for_inference = op->inputIndexes[0];
                for(int32_t input_idx: op->inputIndexes){
                    if(input_idx >= mnn_net_ptr->tensorName.size() || input_idx < 0){
                        std::cout<< "error in tensor input node:" << op->name <<std::endl;
                    }
                    std::string node_input_name = mnn_net_ptr->tensorName[input_idx];
                    node->add_input(node_input_name);
                    auto pre_exist = scope->findValueInfoByName(node_input_name);
                    if(pre_exist == nullptr){
                        // try to find in extraTensorDescribe
                        MNN::TensorDescribeT * tensor_describe = nullptr;
                        if(mnn_net_ptr->extraTensorDescribe.size() > input_idx){
                            tensor_describe = mnn_net_ptr->extraTensorDescribe[input_idx].get();
                        }
                        if(tensor_describe == nullptr){
                            std::cout<< "unable to find input :"<< node_input_name << std::endl;
                        }else{
                            std::vector<int> node_input_dims ;
                            for(int j = 0 ; j < tensor_describe->blob->dims.size(); j++){
                                node_input_dims.push_back(tensor_describe->blob->dims[j]);
                            }
                            onnx::TensorProto_DataType node_input_dtype = OPConvertor::convertDataType(tensor_describe->blob->dataType);
                            auto node_input = OPConvertor::valueInfoProtoFromArgs(node_input_name, node_input_dtype, node_input_dims.size(), node_input_dims);
                            scope->pushValue(node_input);
                        }
                    }
                }
                // run op converter
                OPConvertorManager::sharedInstance()->convertToNode(op, node, scope);
                // output 
                for(int32_t output_idx : op->outputIndexes){
                    if(output_idx >= mnn_net_ptr->tensorName.size() || output_idx < 0){
                        std::cout<< "error in tensor output node:" << op->name <<std::endl;
                    }
                    std::string node_output_name = mnn_net_ptr->tensorName[output_idx];
                    node->add_input(node_output_name);
                    // find extraTensorDescribe First
                    MNN::TensorDescribeT * tensor_describe = nullptr;
                    if(mnn_net_ptr->extraTensorDescribe.size() > output_idx){
                        tensor_describe = mnn_net_ptr->extraTensorDescribe[output_idx].get();
                    }
                    if(tensor_describe == nullptr){
                        // do the shape inference 
                        // wait to add 
                    }else{
                        std::vector<int> node_output_dims ;
                        for(int j = 0 ; j < tensor_describe->blob->dims.size(); j++){
                            node_output_dims.push_back(tensor_describe->blob->dims[j]);
                        }
                        onnx::TensorProto_DataType node_output_dtype = OPConvertor::convertDataType(tensor_describe->blob->dataType);
                        auto node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name, node_output_dtype, node_output_dims.size(), node_output_dims);
                        scope->pushValue(node_output);
                        // check if it is the model output
                        if(std::find(mnn_net_ptr->outputName.begin(), mnn_net_ptr->outputName.end(), node_output_name)!=mnn_net_ptr->outputName.end()){
                            scope->pushOutput(node_output);
                        }
                    }
                }
                scope->pushNode(node);
                OPConvertorManager::sharedInstance()->convertAfter(op, node, scope);
            }
        }
        scope->buildGraph();
    }

    void Convertor::writeFile(std::string &output_path){
        onnx_model->set_producer_name("mnn2onnx");
        onnx_model->mutable_graph()->set_name(mnn_net_ptr->bizCode);
        onnx_model->set_ir_version(9);
        onnx_model->add_opset_import()->set_version(13);
        std::ofstream fs(output_path);
        if(fs.fail()){
            std::cout<<"open failed : "<< output_path << std::endl;
        }else{
            onnx_model->SerializeToOstream(&fs);
        }
        fs.close();
    }
}