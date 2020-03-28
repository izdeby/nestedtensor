#include <seg_layers.h>

namespace torch {
namespace nested_tensor {
    // relu
    THPNestedTensor relu_cpp(THPNestedTensor input, bool inplace=false) {
        return NestedTensor(torch::relu(*input.data().get_buffer()), input.data().nested_size());
    }

    THPNestedTensor relu_out_cpp(THPNestedTensor& input) {
        input = relu_cpp(input, true);
        return input;
    }

    void add_relu(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), relu_cpp);
            m.def((name + std::string("_")).c_str(), relu_out_cpp);
    }

    // dropout
    THPNestedTensor dropout_cpp(THPNestedTensor& input, double p, bool training, bool inplace) {
        if (!inplace) {
            return NestedTensor(torch::dropout(*input.data().get_buffer(), p, training), 
                                input.data().nested_size());
        } else {
            input = NestedTensor(torch::dropout(*input.data().get_buffer(), p, training), 
                                 input.data().nested_size());
            return input;
        }
    }

    void add_dropout(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), dropout_cpp,
                   py::arg("input") = torch::Tensor(), // TODO: update
                   py::arg("p") = 0.5,
                   py::arg("training") = true,
                   py::arg("inplace") = false);
        }

    // conv2d
    THPNestedTensor conv2d_cpp(THPNestedTensor input, 
                               const Tensor& weight, 
                               const Tensor& bias = torch::Tensor(), 
                               int64_t stride=1,
                               int64_t padding=0,
                               int64_t dilation=1,
                               int64_t groups=1) {
        
    }

    void add_conv2d(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
        m.def(name.c_str(), conv2d_cpp, "bla bla", 
            py::arg("input") = torch::Tensor(),  // TODO: update
            py::arg("weight") = torch::Tensor(), // TODO: update
            py::arg("bias") = torch::Tensor(),   // TODO: update
            py::arg("stride") = 1,
            py::arg("padding") = 0,
            py::arg("dilation") = 1,
            py::arg("groups") = 1);
        //m.def((name + std::string("_")).c_str(), conv2d_out_cpp);
    }
}
}