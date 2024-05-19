#include <string>
#include "MNN_generated.h"
#include "onnx.pb.h"
#include "vector"

namespace LLX {
    class ConvertorScope
    {
    private:
        MNN::NetT *net_t;
        onnx::GraphProto *graph_proto;

        std::vector<onnx::TensorProto *> initializiers;
        std::vector<onnx::ValueInfoProto*> inputs;
        std::vector<onnx::ValueInfoProto*> outputs;
        std::vector<onnx::NodeProto *> nodes;
        std::vector<std::string> value_names;
        std::vector<onnx::ValueInfoProto *>values;
    public:
        ConvertorScope() = delete;
        ConvertorScope(MNN::NetT *net , onnx::GraphProto *proto){
            net_t = net;
            graph_proto = proto;
        };
        ~ConvertorScope();
        onnx::NodeProto * findNodeByName(std::string & name);
        onnx::ValueInfoProto *findValueInfoByName(std::string &name);
        onnx::TensorProto *findInitializerByName(std::string &name);
        void pushInitializer(onnx::TensorProto * initializer);
        void pushInput(onnx::ValueInfoProto *input);
        void pushOutput(onnx::ValueInfoProto *output);
        void pushNode(onnx::NodeProto *node);
        void pushValue(onnx::ValueInfoProto *val);
        void buildGraph();
    };
}