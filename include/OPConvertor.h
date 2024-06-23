#include "MNN_generated.h"
#include "onnx.pb.h"
#include "Tensor_generated.h"
#include "Type_generated.h"
#include "ConvertorScope.h"
#include <map>

namespace LLX{
    class OPConvertor
    {
    private:
        /* data */
    public:
        virtual void run(onnx::NodeProto *onnx_node, MNN::OpT *src_op, ConvertorScope *scope)=0;
        virtual ~OPConvertor(){};
        static onnx::TensorProto_DataType convertDataType(MNN::DataType type);
        static onnx::TensorProto * convertBlobToTensor(const MNN::BlobT *blob);
        static onnx::ValueInfoProto * valueInfoProtoFromInitialzer(onnx::TensorProto * tensor);
        static onnx::TensorProto * tensorProtoFromOp(MNN::OpT *op);
        static onnx::ValueInfoProto * valueInfoProtoFromArgs(const std::string &name,  int type, int dim_size , std::vector<int> dims );
    };
    
    class OPConvertorManager{
    public:
        static OPConvertorManager *sharedInstance();
        OPConvertorManager();
        void registerNodeConvertor(MNN::OpType op_type, OPConvertor *OPConvertor);
        void convertToNode(MNN::OpT *op, onnx::NodeProto* node, ConvertorScope *scope);
        void convertAfter(MNN::OpT *op, onnx::NodeProto *node, ConvertorScope *scope);
    private:
        static OPConvertorManager *manager;
        std::map<MNN::OpType , OPConvertor *>convertor_map;
    };

#define DECLARE_OP_CONVERTOR(OPName)        \
class OPName : public OPConvertor {         \
public:                                     \
    void run(onnx::NodeProto *onnx_node, MNN::OpT * src_op, ConvertorScope *scope);   \
    ~OPName(){};                            \
};

#define REGISTER_OP_CONVERTOR(OPType, OPName)   \
registerNodeConvertor(OPType, new OPName());

#pragma mark op class declaration
DECLARE_OP_CONVERTOR(ArgMaxOPConvertor)
DECLARE_OP_CONVERTOR(ArgMinOPConvertor)
DECLARE_OP_CONVERTOR(BinaryOPConvertor)
DECLARE_OP_CONVERTOR(ConvolutionDepthwiseOPConvertor)
DECLARE_OP_CONVERTOR(ConvolutionOPConvertor)
}