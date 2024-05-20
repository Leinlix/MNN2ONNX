#include "Convertor.h"
#include "OPConvertor.h"

#include <iostream>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <fstream>

namespace LLX{
    onnx::TensorProto * makeInitiailizer(MNN::OpT *op){
        auto op_param = op->main.value;
        onnx::TensorProto *init = LLX::OPConvertor::convertBlobToTensor((MNN::BlobT *)op_param);
        init->set_name(op->name);
        return init;
    }

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
        // wait to add 
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