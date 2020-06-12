#pragma once
#include <nestedtensor/csrc/utils/nested_node.h>
#include <ATen/ATen.h>

namespace torch {
namespace nested_tensor {

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;
using SizeNode = NestedNode<c10::List<int64_t>>;
using IntegerNode = NestedNode<int64_t>;

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct NestedTensor {
  NestedTensor() = delete;
  NestedTensor(TensorNode&& structure);
  NestedTensor(at::Tensor&& buffer, TensorNode&& structure);
  NestedTensor(at::Tensor&& buffer, SizeNode nested_size);
  c10::optional<at::Tensor>& get_buffer() {
    return _buffer;
  }
  const c10::optional<at::Tensor>& get_buffer() const {
    return _buffer;
  }
  std::vector<c10::optional<int64_t>> sizes() const;
  caffe2::TypeMeta dtype() const {
    return _first_variable.dtype();
  }
  int64_t element_size() const {
    return _first_variable.element_size();
  }
  // This is a C++ representation of a nested list of torch.Sizes
  //
  // It can never be a list of just numbers, because torch.Size
  // is always a list and NestedTensors represent lists of torch.Tensors
  //
  // Noteworthy cases:
  //
  // This is an empty list of lists if we construct
  // nested_tensor([])
  // which is of nested_dim 1, dim 1 and tensor_dim 0
  //
  // This is a list of empty lists if we construct e.g.
  // nested_tensor([torch.tensor(0), torch.tensor(1), ...])
  // which is of nested_dim 1, dim 1 and tensor_dim 0
  //
  // This is a list of list of numbers if we construct e.g.
  // nested_tensor([torch.tensor([1]), torch.tensor([2]), ...])
  // which is of nested_dim 1, dim 2 and tensor_dim 1
  //
  // That means, if the list is not empty it is either a list of
  // lists of numbers or a list of empty lists.
  SizeNode nested_size() const {
    return _nested_size;
  }
  SizeNode nested_stride() const {
    return map(
        [](at::Tensor tensor) { return c10::List<int64_t>(tensor.strides()); },
        _structure);
  }
  NestedTensor pin_memory() {
    // NOTE: The assumption here is that pin_memory will materialize
    // the views that _structure contains when NestedTensor is contiguous.
    return NestedTensor(
        map([](at::Tensor tensor) { return tensor.pin_memory(); }, _structure));
  }
  NestedTensor grad() {
    return NestedTensor(
        map([](at::Tensor tensor) { return tensor.grad(); }, _structure));
  }
  NestedTensor detach() {
    // NOTE: For the contiguous case the tensors in _structure are views
    // of parts of _buffer and the returned detached views will still
    // modify that buffer if using in-place methods etc.
    return NestedTensor(
        map([](at::Tensor tensor) { return tensor.detach(); }, _structure));
  }
  NestedTensor requires_grad_(bool requires_grad) {
    apply(
        [requires_grad](at::Tensor tensor) -> void {
          tensor.set_requires_grad(requires_grad);
        },
        _structure);
    if (is_contiguous()) {
      (*_buffer).set_requires_grad(requires_grad);
    }
    return *this;
  }
  void backward(NestedTensor gradient, bool retain_graph, bool create_graph) {
    if (is_contiguous() && gradient.is_contiguous()) {
      (*_buffer).backward((*gradient.get_buffer()), retain_graph, create_graph);
    } else {
      apply(
          [retain_graph, create_graph](
              at::Tensor tensor1, at::Tensor tensor2) -> void {
            tensor1.backward(tensor2, retain_graph, create_graph);
          },
          _structure,
          gradient.get_structure());
    }
  }
  int64_t __len__() const {
    return _structure.degree();
  }
  at::Tensor to_tensor();
  NestedTensor to_nested_tensor(c10::optional<int64_t> dim);
  int64_t nested_dim() const {
    return _structure.height();
  }
  at::ScalarType scalar_type() const {
    return _first_variable.scalar_type();
  }
  at::Backend backend() const {
    return options().backend();
  }
  at::Layout layout() const {
    return _first_variable.layout();
  }
  at::Device device() const {
    return _first_variable.device();
  }
  at::TensorOptions options() const {
    return _first_variable.options();
  }
  bool requires_grad() const {
    return _first_variable.requires_grad();
  }
  int64_t dim() const {
    return _first_variable.dim() + nested_dim();
  }
  int64_t numel() const {
    auto fn = [](at::Tensor leaf, int64_t input) {
      return input + leaf.numel();
    };
    return reduce<decltype(fn), int64_t, at::Tensor>(_structure, fn, 0);
  }
  bool is_pinned() const {
    if (is_contiguous()) {
      return (*_buffer).is_pinned();
    } else {
      return _first_variable.is_pinned();
    }
  }
  bool is_contiguous() const {
    // NOTE: The Tensors themselves might not be contiguous even if there is a
    // buffer. For this to be contiguous not only the individuals Tensors have
    // to be but also the buffer.
    auto fn = [](at::Tensor leaf, bool input) {
      return input && leaf.is_contiguous();
    };
    return _buffer && (*_buffer).is_contiguous() &&
        reduce<decltype(fn), bool, at::Tensor>(_structure, fn, true);
  }
  NestedTensor contiguous() const;
  TensorNode& get_structure() {
    return _structure;
  }
  const TensorNode& get_structure() const {
    return _structure;
  }

// torch.Tensor methods
  NestedTensor copy_(const NestedTensor& source, bool non_blocking=false);
  NestedTensor squeeze_(c10::optional<int64_t> dim);

 private:
  c10::optional<at::Tensor> _buffer;
  TensorNode _structure;
  at::Tensor _first_variable;
  SizeNode _nested_size;
};

} // namespace nested_tensor
} // namespace torch

namespace at {

constexpr auto NestedTensorKey = DispatchKey::PrivateUse1_PreAutograd;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(torch::nested_tensor::NestedTensor&& data)
      : TensorImpl(
            c10::DispatchKeySet(NestedTensorKey),
            data.dtype(),
            data.device()),
        _data(std::move(data)) {
            for (auto opt_int : _data.sizes()) {
              if (opt_int) {
                _sizes.push_back(*opt_int);
              }
            }
        }

  int64_t dim() const override {
    return _data.dim();
  }
  int64_t numel() const override {
    return _data.numel();
  }
  bool is_contiguous(
      at::MemoryFormat memory_format) const override {
    return _data.is_contiguous();
  }

  IntArrayRef sizes() const override;
  int64_t size(int64_t dim) const override;
  IntArrayRef strides() const override;

  torch::nested_tensor::NestedTensor _data;
  std::vector<int64_t> _sizes;
};

inline bool is_nested_tensor_impl(const at::Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey);
}

inline at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor) {
  if (!is_nested_tensor_impl(tensor)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  return static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline torch::nested_tensor::NestedTensor get_nested_tensor(
    const at::Tensor tensor) {
  return get_nested_tensor_impl(tensor)->_data;
}

inline torch::nested_tensor::TensorNode get_nested_tensor_structure(
    const at::Tensor tensor) {
  return get_nested_tensor(tensor).get_structure();
}

inline bool is_tensor_shape(const at::Tensor tensor) {
  auto nt = get_nested_tensor(tensor);
  for (const auto& size : nt.sizes()) {
    if (!size) {
      return false;
    }
  }
  return true;
}

inline at::Tensor wrap_nested_tensor(
    torch::nested_tensor::NestedTensor&& result) {
  return at::detail::make_tensor<NestedTensorImpl>(std::move(result));
}

inline at::Tensor wrap_tensor_node(
    torch::nested_tensor::TensorNode&& result) {
  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(result)));
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_);

inline std::ostream& operator<<(std::ostream& out, const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor._data.get_structure();
  out << "NESTED_TENSOR";
  apply([&out](at::Tensor tensor) { out << tensor << std::endl; }, node);
  out << std::endl;
  return out;
}

}
