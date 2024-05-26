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
                    node->add_output(node_output_name);
                    // find extraTensorDescribe First
                    MNN::TensorDescribeT * tensor_describe = nullptr;
                    if(mnn_net_ptr->extraTensorDescribe.size() > output_idx){
                        tensor_describe = mnn_net_ptr->extraTensorDescribe[output_idx].get();
                    }
                    if(tensor_describe == nullptr){
                        // do the shape inference 
                        onnx::ValueInfoProto *p = scope->findValueInfoByName(mnn_net_ptr->tensorName[input_idx_for_inference]);
                        if(p!=nullptr){
                            onnx::ValueInfoProto *node_output;
                            if(op->type == MNN::OpType_Convolution || op->type == MNN::OpType_ConvolutionDepthwise){
                                MNN::Convolution2DT * op_param = (MNN::Convolution2DT *)op->main.value;
                                 std::vector<int> di;
                                for(int j = 0; j < 4; j++){
                                    int dim = p->type().tensor_type().shape().dim()[j].dim_value();
                                    if(j == 1){
                                        dim = op_param->common->outputCount;
                                    }else if(j==2){
                                        dim = (dim + 2* op_param->common->padX + op_param->common->strideX - ((op_param->common->kernelX -1)*op_param->common->dilateX + 1))/op_param->common->strideX;
                                    }else if (j == 3){
                                        dim = (dim + 2* op_param->common->padY + op_param->common->strideY - ((op_param->common->kernelY -1)*op_param->common->dilateY + 1))/op_param->common->strideY;
                                    }
                                    di.push_back(dim);
                                }
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name,p->type().tensor_type().elem_type(),di.size(),di);
                            }else if (op->type == MNN::OpType_Pooling){
                                MNN::PoolT * op_param = (MNN::PoolT *)op->main.value;
                                std::vector<int> di;
                                for(int j = 0; j < 4; j++){
                                    int dim = p->type().tensor_type().shape().dim()[j].dim_value();
                                    if(j==2){
                                        dim = (dim + op_param->strideX - op_param->kernelX)/op_param->strideX;
                                    }else if (j == 3){
                                        dim = (dim + op_param->strideY - op_param->kernelY)/op_param->strideY;
                                    }
                                    di.push_back(dim);
                                }
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name,p->type().tensor_type().elem_type(),di.size(),di);
                            }else if (op->type == MNN::OpType_Pooling3D){
                                std::vector<int> di;
                                for(int j = 0; j < 4; j++){
                                    int dim = p->type().tensor_type().shape().dim()[j].dim_value();
                                    if(j>1){
                                        dim = 1;
                                    }
                                    di.push_back(dim);
                                }
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name,p->type().tensor_type().elem_type(),di.size(),di);
                            }else if(op->type == MNN::OpType_Interp){
                                auto input_2_init = scope->findInitializerByName(mnn_net_ptr->tensorName[op->inputIndexes[1]]);
                                if (input_2_init != nullptr){
                                    input_2_init->set_data_type(onnx::TensorProto_DataType_INT64);
                                    std::vector<int> di;
                                    // change data to int64 and set value
                                    for(int j = 0 ; j < input_2_init->dims_size(); j++){
                                        for(int k = 0 ; k < input_2_init->dims(j);k++){
                                            input_2_init->add_int64_data(input_2_init->int32_data(k));
                                            di.push_back(input_2_init->int32_data(k));
                                        }
                                    }
                                    input_2_init->clear_int32_data();
                                    node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name,p->type().tensor_type().elem_type(),di.size(),di);
                                }else{
                                    std::cout<<"error in shape inferece for interp, can't find input value"<<std::endl;
                                }
                            }else if (op->type == MNN::OpType_Reshape){
                                auto input_2_init = scope->findInitializerByName(mnn_net_ptr->tensorName[op->inputIndexes[1]]);
                                if (input_2_init != nullptr)
                                {
                                    input_2_init->set_data_type(onnx::TensorProto_DataType_INT64);
                                    int64_t all = 1;
                                    for(int j = 0 ; j < p->type().tensor_type().shape().dim_size(); j++){
                                        all *= p->type().tensor_type().shape().dim()[j].dim_value();
                                    }
                                    // change data to int64
                                    for(int j = 0 ; j < input_2_init->dims_size(); j++){
                                        for(int k = 0 ; k < input_2_init->dims(j);k++){
                                            input_2_init->add_int64_data(input_2_init->int32_data(k));
                                            if(input_2_init->int32_data(k)!=-1){
                                                all /= input_2_init->int32_data(k);
                                            }
                                        }
                                    }
                                    input_2_init->clear_int32_data();
                                    std::vector<int> di;
                                    for(int j = 0 ; j< input_2_init->int64_data_size(); j++){
                                        if(input_2_init->int64_data(j) == -1){
                                            di.push_back(all);
                                        }else{
                                            di.push_back(input_2_init->int64_data(j));
                                        }
                                    }
                                    node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name,p->type().tensor_type().elem_type(),di.size(),di);
                                }else{
                                    std::cout<<"error in shape inferece for reshap, can't find input value"<<std::endl;
                                }
                            }else if(op->type == MNN::OpType_Concat){
                                MNN::AxisT *axis = (MNN::AxisT *)op->main.value;
                                auto input_2 = scope->findValueInfoByName(mnn_net_ptr->tensorName[op->inputIndexes[1]]); 
                                std::vector<int>di ;
                                for(int j = 0 ; j < input_2->type().tensor_type().shape().dim_size(); j++){
                                    if(j == axis->axis){
                                        di.push_back(input_2->type().tensor_type().shape().dim()[j].dim_value() + p->type().tensor_type().shape().dim()[j].dim_value() )
                                    }else{
                                        di.push_back(input_2->type().tensor_type().shape().dim()[j].dim_value());
                                    }
                                }
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name,p->type().tensor_type().elem_type(),di.size(),di);
                            }else if(op->type == MNN::OpType_ArgMax){
                                std::vector<int> di;
                                for(int j = 0 ; j < p->type().tensor_type().shape().dim_size(); j++){
                                    di.push_back(p->type().tensor_type().shape().dim()[j].dim_value());
                                }
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name, onnx::TensorProto_DataType_INT64, di.size(),di);
                            }else if(op->type == MNN::OpType_Rank){ 
                                std::vector<int> di ;
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name, onnx::TensorProto_DataType_INT64, di.size(),di);
                            }else if (op->type == MNN::OpType_Shape){
                                std::vector<int> di ;
                                di.push_back(p->type().tensor_type().shape().dim_size());
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name, onnx::TensorProto_DataType_INT64, di.size(),di);
                            }else if(op->type == MNN::OpType_Flatten){
                                MNN::FlattenT * op_param = (MNN::FlattenT*)op->main.value;
                                int dim1 = 1, dim2 = 1;
                                std::vector<int> di;
                                for(int j = 0; j < p->type().tensor_type().shape().dim_size();j++){
                                    if(j < op_param->axis){
                                        dim1 *= p->type().tensor_type().shape().dim(j).dim_value();
                                    }else {
                                        dim2 *= p->type().tensor_type().shape().dim(j).dim_value();
                                    }
                                }
                                di.push_back(dim1);
                                di.push_back(dim2);
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name,onnx::TensorProto_DataType_INT64,di.size(),di);
                            }else{
                                std::vector<int> di;
                                for(int j = 0 ; j < p->type().tensor_type().shape().dim_size(); j++){
                                    di.push_back(p->type().tensor_type().shape().dim()[j].dim_value());
                                }
                                node_output = OPConvertor::valueInfoProtoFromArgs(node_output_name,p->type().tensor_type().elem_type(),di.size(),di);
                            }
                            scope->pushValue(node_output);
                            // check if it is the model output
                            if(std::find(mnn_net_ptr->outputName.begin(), mnn_net_ptr->outputName.end(), node_output_name)!=mnn_net_ptr->outputName.end()){
                                scope->pushOutput(node_output);
                            }
                        }else{
                            std::cout<<"unable inference the output node with no input!"<<std::endl;
                        }
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