#include <ATen/WrapDimUtils.h>
#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/Size.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

py::object _nested_helper(
    c10::optional<int64_t> index,
    SizeNode&& size_node) {
  auto fn = [](auto& self, const SizeNode& s, int64_t dim) -> py::object {
    if (dim == 0) {
      return py::cast(s.degree());
    }
    // List of Tensors
    if (s.height() == 1) {
      std::vector<int64_t> result;
      for (const auto& child : s.unbind()) {
        result.push_back(child.payload().get(dim - 1));
      }
      return py::tuple(py::cast(result));
    }
    std::vector<py::object> result;
    for (const auto& child : s.unbind()) {
      result.emplace_back(self(self, child, dim - 1));
    }
    return py::tuple(py::cast(result));
  };
  return fn(fn, size_node, *index);
}

py::object THPNestedTensor::nested_size(c10::optional<int64_t> index_) {
  if (!index_) {
    return py::cast(THPPythonNode(
        map([](c10::List<int64_t> e)
          { 
          std::vector<int64_t> e_vec = e.vec();
          return py::reinterpret_steal<py::object>(
              THPSize_NewFromSizes(e_vec.size(), e_vec.data()));
          },
          _data.nested_size()),
        "NestedSize"));
  }
  int64_t index = at::maybe_wrap_dim((*index_), _data.dim());
  SizeNode size_node = _data.nested_size();
  return _nested_helper(index, std::move(size_node));
}

py::object THPNestedTensor::nested_stride(c10::optional<int64_t> index_) {
  if (!index_) {
    return py::cast(THPPythonNode(
        map([](c10::List<int64_t> e) -> py::object
          {
          return py::tuple(py::cast(e.vec()));
          },
          _data.nested_stride()),
        "NestedStride"));
  }
  int64_t index = at::maybe_wrap_dim((*index_), _data.dim());
  SizeNode size_node = _data.nested_stride();
  return _nested_helper(index, std::move(size_node));
}

std::string THPNestedTensor::str() {
  auto node = _data.get_structure();
  return NestedNode___str__(
      node, "nested_tensor", [](c10::IValue payload, const std::string& tabs) {
        std::vector<std::string> tokens = split_str(
            THPUtils_unpackString(
                PyObject_Str(THPVariable_Wrap(payload.toTensor()))),
            "\n");
        std::string result;
        for (size_t i = 0; i < tokens.size(); i++) {
          result = result + tabs + tokens[i];
          if (i < tokens.size() - 1) {
            result = result + "\n";
          }
        }
        return result;
      });
}

py::object THPNestedTensor::getDtype() {
  return py::reinterpret_steal<py::object>(
      torch::autograd::utils::wrap(torch::getTHPDtype(_data.scalar_type())));
}

py::object THPNestedTensor::getLayout() {
  return py::reinterpret_steal<py::object>(
      torch::autograd::utils::wrap(torch::getTHPLayout(_data.layout())));
}

py::object THPNestedTensor::getDevice() {
  return torch::jit::toPyObject(_data.device());
}

// TODO: Can't have mixed return types in C++
// for different input values.
py::object THPNestedTensor::to_tensor(c10::optional<int64_t> dim_) {
  if (!dim_) {
    return py::cast(_data.to_tensor());
  }
  int64_t dim = at::maybe_wrap_dim((*dim_), _data.dim());
  if (dim == 0) {
    return py::cast(_data.to_tensor());
  }
  // If dim is bigger than nested_dim the NestedTensor is already
  // of Tensor for dimensions bigger than the given.
  if (nested_dim() == 1) {
    return py::cast(*this);
  }
  // At this point nested_dim is at least 2. That means any unbind
  // operation of a child must yield NestedTensors.
  // If dim is 1 then we'll apply to_tensor(0) to the children and must expect
  // Tensors.
  std::vector<TensorNode> result;
  if (dim == 1) {
    for (py::object child : unbind(0)) {
      result.push_back(TensorNode(
          py::cast<at::Tensor>(py::cast<THPNestedTensor>(child).to_tensor(0))));
    }
  } else {
    for (py::object child : unbind(0)) {
      result.push_back(py::cast<THPNestedTensor>(
                           py::cast<THPNestedTensor>(child).to_tensor(dim - 1))
                           ._data.get_structure());
    }
  }
  return py::cast(THPNestedTensor(TensorNode(std::move(result))));
}

// TODO: Since this returns vies there is no reason not to return a sequence
// of contiguous NestedTensors for a given NestedTensor.
std::vector<py::object> THPNestedTensor::unbind(int64_t dim) {
  dim = at::maybe_wrap_dim(dim, _data.dim());
  auto node = _data.get_structure();
  auto nested_dim = _data.nested_dim();
  if (nested_dim == 1) {
    if (dim == 0) {
      std::vector<py::object> result;
      for (const auto& child : node.unbind()) {
        result.push_back(py::cast(child.payload()));
      }
      return result;
    } else {
      int64_t dim_max_size = 0;
      for (const auto& child : node.unbind()) {
        int64_t dim_size = child.payload().size(dim - 1);
        dim_max_size = dim_max_size > dim_size ? dim_max_size : dim_size;
      }
      std::vector<std::vector<TensorNode>> unbound;
      unbound.resize(dim_max_size);
      for (const auto& child : node.unbind()) {
        std::vector<at::Tensor> unbound_tensors =
            at::unbind(child.payload(), dim - 1);
        for (size_t i = 0; i < unbound_tensors.size(); i++) {
          unbound[i].push_back(TensorNode(std::move(unbound_tensors[i])));
        }
      }
      std::vector<py::object> result;
      for (size_t i = 0; i < unbound.size(); i++) {
        TensorNode tmp = TensorNode(std::move(unbound[i]));
        THPNestedTensor tmp_thd = THPNestedTensor(NestedTensor(std::move(tmp)));
        result.push_back(py::cast(tmp_thd));
      }
      return result;
    }
  }
  std::vector<THPNestedTensor> unbound_thp;
  for (auto child : node.unbind()) {
    unbound_thp.push_back(THPNestedTensor(NestedTensor(std::move(child))));
  }
  if (dim == 0) {
    std::vector<py::object> result;
    for (auto elem : unbound_thp) {
      result.push_back(py::cast(elem));
    }
    return result;
  }
  std::vector<std::vector<TensorNode>> unbound;
  for (size_t i = 0; i < unbound_thp.size(); i++) {
    std::vector<py::object> tmp = unbound_thp[i].unbind(dim - 1);
    for (size_t j = 0; j < tmp.size(); j++) {
      if (unbound.size() >= j) {
        unbound.resize(j + 1);
      }
      py::object tmp_j = tmp[j];
      if (py::isinstance<THPNestedTensor>(tmp_j)) {
        unbound[j].push_back(
            py::cast<THPNestedTensor>(tmp[j])._data.get_structure());
      } else {
        unbound[j].push_back(TensorNode(py::cast<at::Tensor>(tmp_j)));
      }
    }
  }
  std::vector<py::object> result;
  for (size_t i = 0; i < unbound.size(); i++) {
    TensorNode tmp = TensorNode(std::move(unbound[i]));
    THPNestedTensor tmp_thd = THPNestedTensor(NestedTensor(std::move(tmp)));
    result.push_back(py::cast(tmp_thd));
  }
  return result;
}

} // namespace nested_tensor
} // namespace torch
