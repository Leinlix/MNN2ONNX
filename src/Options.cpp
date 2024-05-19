#include "cxxopts.hpp"
#include "Convertor.h"

int main (int argc , char *argv[]){
    std::string input_path , output_path;

    cxxopts::Options options("MNN2ONNX");
    options.positional_help("[optional args]").show_positional_help();
    options.allow_unrecognised_options().add_options()(
        std::make_pair("help","h"),
        "Convert MNN modle to ONNX model./n"
    )(
        std::make_pair("input", "i"),
        "the input: MNN model path",
        cxxopts::value<std::string>()
    )(
        std::make_pair("output", "o"),
        "the output: ONNX model output path",
        cxxopts::value<std::string>()
    );

    auto result = options.parse(argc, argv);
    if(result.count("help")){
        std::cout<<options.help({""})<<std::endl;
        return 0;
    }
    if(result.count("input")){
        input_path = result["input"].as<std::string>();
    }else{
        std::cout<<"Must give the input path!"<<std::endl;
    }
    if(result.count("output")){
        input_path = result["output"].as<std::string>();
    }else{
        std::cout<<"Must give the output path!"<<std::endl;
    }

    LLX::Convertor convertor;
    convertor.loadModel(input_path);
    convertor.convertModel();
    convertor.writeFile(output_path);
    return 0;
}