/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <torch/extension.h>

#include <sox.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "speech_augment.h"

namespace py_wavaugment {
std::vector<std::string> get_effect_names() {
  auto effects = cpp_sox::get_supported_effects();
  std::vector<std::string> result(effects.begin(), effects.end());
  return result;
}

sox_signalinfo_t dict2signalinfo(const py::dict &d) {
  sox_signalinfo_t signal{.rate = 16000,
                          .channels = 1,
                          .precision = 32,
                          .length = 0,
                          .mult = nullptr};
  if (d.contains("rate")) {
    auto rate = d["rate"];
    if (py::isinstance<py::int_>(rate)) {
      signal.rate = rate.cast<int32_t>();
    } else if (py::isinstance<py::float_>(rate)) {
      signal.rate = rate.cast<float_t>();
    } else {
      throw std::runtime_error("rate must be float or int");
    }
  }
  if (d.contains("channels")) {
    auto channels = d["channels"];
    if (py::isinstance<py::int_>(channels)) {
      signal.channels = channels.cast<int32_t>();
    } else {
      throw std::runtime_error("channels must be int");
    }
  }
  if (d.contains("length")) {
    auto length = d["length"];
    if (py::isinstance<py::int_>(length)) {
      signal.length = length.cast<int32_t>();
    } else {
      throw std::runtime_error("length must be int");
    }
  }
  return signal;
}

int shutdown_sox() {
  /* Shutdown for sox effects.  Do not shutdown between multiple calls  */
  return sox_quit();
}

class PyEffectChain {
  cpp_sox::EffectChain chain;

public:
  void add_effect(const std::string &effect_name,
                  const std::vector<std::string> &effect_params) {
    chain.add_effect(effect_name, effect_params);
  }

  int apply_flow_effects(at::Tensor itensor, at::Tensor otensor,
                         const py::dict &src_info,
                         const py::dict &target_info) {
    sox_signalinfo_t src_signal = dict2signalinfo(src_info);
    sox_encodinginfo_t src_encoding = default_encoding;

    sox_signalinfo_t dst_signal = dict2signalinfo(target_info);
    sox_encodinginfo_t dst_encoding = default_encoding;

    std::vector<sox_sample_t> src(itensor.numel());
    AT_DISPATCH_ALL_TYPES(itensor.scalar_type(), "write_audio_buffer", [&] {
      auto *data = itensor.data_ptr<scalar_t>();
      std::copy(data, data + itensor.numel(), src.begin());
    });

    std::vector<sox_sample_t> dst(src.size());

    chain.apply(src, src_signal, src_encoding, dst, dst_signal, dst_encoding);

    int ns = dst.size();
    int nc = dst_signal.channels;

    otensor.resize_({ns / nc, nc});
    otensor = otensor.contiguous();

    AT_DISPATCH_ALL_TYPES(otensor.scalar_type(), "effects_buffer", [&] {
      auto *data = otensor.data_ptr<scalar_t>();
      std::copy(dst.begin(), dst.end(), data);
    });

    otensor.transpose_(1, 0);
    int sr = dst_signal.rate;
    return sr;
  }

private:
  const sox_encodinginfo_t default_encoding{.encoding = SOX_ENCODING_SIGN2,
                                            .bits_per_sample = 32,
                                            .compression = 0,
                                            .reverse_bytes = sox_option_default,
                                            .reverse_nibbles =
                                                sox_option_default,
                                            .reverse_bits = sox_option_default,
                                            .opposite_endian = sox_false};
};

} // namespace py_wavaugment

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<py_wavaugment::PyEffectChain>(m, "PyEffectChain",
                                           py::module_local())
      .def(py::init<>())
      .def("add_effect", &py_wavaugment::PyEffectChain::add_effect)
      .def("apply_flow_effects",
           &py_wavaugment::PyEffectChain::apply_flow_effects);
  m.def("get_effect_names", &py_wavaugment::get_effect_names,
        "supported effects");
  m.def("shutdown_sox", &py_wavaugment::shutdown_sox,
        "shutdown sox for effects");
}
