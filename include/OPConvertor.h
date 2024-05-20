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
    };
    
    class OPConvertorManager{
    public:
        static OPConvertor *sharedInstance();
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
registerNodeConvertor(OPTyPE, new OPName());

#pragma mark op class declaration

}