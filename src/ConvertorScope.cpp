#include "ConvertorScope.h"

namespace LLX {
    onnx::NodeProto * ConvertorScope::findNodeByName(std::string &name){
        onnx::NodeProto * ret = nullptr;
        for(int i = 0 ; i < nodes.size(); i++){
            if(nodes[i]->name() == name){
                ret = nodes[i];
                break;
            }
        }
        return ret;
    }

    onnx::ValueInfoProto * ConvertorScope::findValueInfoByName(std::string &name){
        onnx::ValueInfoProto *ret = nullptr;
        bool find = false;
        for(int i = 0; i <inputs.size(); i++){
            if(inputs[i]->name() == name){
                ret = inputs[i];
                find = true;
                break;
            }
        }
        if(find){
            return ret;
        }
        for(int i = 0; i <values.size(); i++){
            if(values[i]->name() == name){
                ret = values[i];
                find = true;
                break;
            }
        }
        if(find){
            return ret;
        }
        for(int i = 0; i <outputs.size(); i++){
            if(outputs[i]->name() == name){
                ret = outputs[i];
                find = true;
                break;
            }
        }
        return ret;
    }

    onnx::TensorProto * ConvertorScope::findInitializerByName(std::string &name){
         onnx::TensorProto * ret = nullptr;
        for(int i = 0 ; i < initializiers.size(); i++){
            if(initializiers[i]->name() == name){
                ret = initializiers[i];
                break;
            }
        }
        return ret;
    }

    void ConvertorScope::pushInitializer(onnx::TensorProto * init){
        initializiers.push_back(init);
    }

    void ConvertorScope::pushInput(onnx::ValueInfoProto *input){
        inputs.push_back(input);
    }

    void ConvertorScope::pushOutput(onnx::ValueInfoProto *output){
        outputs.push_back(output);
    }

    void ConvertorScope::pushNode(onnx::NodeProto *node){
        nodes.push_back(node);
    }

    void ConvertorScope::pushValue(onnx::ValueInfoProto * val){
        if(std::find(value_names.begin() , value_names.end(), val->name())== value_names.end()){
            values.push_back(val);
            value_names.push_back(val->name());
        }
    }

    void ConvertorScope::buildGraph(){
        for(auto init : initializiers){
            graph_proto->mutable_initializer()->AddAllocated(init);
        }
        for(auto input : inputs){
            graph_proto->mutable_input()->AddAllocated(input);
        }
        for(auto output : outputs){
            graph_proto->mutable_output()->AddAllocated(output);
        }
        for(auto value: values){
            graph_proto->mutable_value_info()->AddAllocated(value);
        }
    }
}