#include "ConvertorScope.h"
#include "MNN_generated.h"
#include "onnx.pb.h"
#include <memory.h>
#include <string>
#include "flatbuffers/flatbuffers.h"

namespace LLX{
    class Convertor
    {
    private:
        std::unique_ptr<MNN::NetT> mnn_net_ptr;
        onnx::ModelProto *onnx_model;
        ConvertorScope *scope;
    public:
        void loadModel(std::string &intput_path);
        void convertModel();
        void writeFile(std::string &output_path);  
    };
}