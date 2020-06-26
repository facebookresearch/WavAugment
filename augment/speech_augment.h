/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include <sox.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace cpp_sox {
std::unordered_set<std::string> get_supported_effects();

std::unordered_set<std::string> UNSUPPORTED_EFFECTS = {"spectrogram", "splice",
                                                       "noiseprof", "fir"};
std::unordered_set<std::string> SUPPORTED_EFFECTS = get_supported_effects();

std::unordered_set<std::string> get_supported_effects() {
  sox_effect_fn_t const *fns = sox_get_effect_fns();
  std::unordered_set<std::string> effect_names;
  for (int i = 0; fns[i]; ++i) {
    const sox_effect_handler_t *eh = fns[i]();
    if (eh && eh->name &&
        UNSUPPORTED_EFFECTS.find(eh->name) == UNSUPPORTED_EFFECTS.end())
      effect_names.insert(eh->name);
  }
  return effect_names;
};

int init_sox_once() {
  static int result = 0;
  static std::once_flag sox_init_flag;

  std::call_once(sox_init_flag, [&]() { result = sox_init(); });
  return result;
}

// https://stackoverflow.com/a/43626234
struct free_deleter {
  template <typename T> void operator()(T *p) const { std::free(p); }
};
template <typename T> using unique_c_ptr = std::unique_ptr<T, free_deleter>;
using unique_sox_effect_ptr = unique_c_ptr<sox_effect_t>;

class EffectChain {
  std::vector<unique_sox_effect_ptr> effects;
  EffectChain(const EffectChain &) = delete;

public:
  void apply(sox_format_t *input, sox_format_t *output) {
    sox_effects_chain_t *chain =
        sox_create_effects_chain(&input->encoding, &output->encoding);

    // build up chain, with input and output
    unique_sox_effect_ptr input_effect(sox_create_effect(sox_find_effect("input")));
    char *io_args[1];
    io_args[0] = (char *)input;
    sox_effect_options(input_effect.get(), 1, io_args);

    sox_signalinfo_t interm_signal = input->signal;
    sox_add_effect(chain, input_effect.get(), &interm_signal, &input->signal);

    for (auto &effect : effects) {
      sox_add_effect(chain, effect.get(), &interm_signal, &output->signal);
    }

    unique_sox_effect_ptr output_effect(sox_create_effect(sox_find_effect("output")));
    io_args[0] = (char *)output;
    sox_effect_options(output_effect.get(), 1, io_args);
    sox_add_effect(chain, output_effect.get(), &interm_signal, &output->signal);

    sox_flow_effects(chain, nullptr, nullptr);

    sox_delete_effects_chain(chain);
  }

  void apply(const std::vector<sox_sample_t> &src, sox_signalinfo_t src_signal,
             sox_encodinginfo_t src_encoding, std::vector<sox_sample_t> &dst,
             sox_signalinfo_t &dst_signal, sox_encodinginfo_t &dst_encoding) {

    auto file_type = "raw";

    char *buffer;
    size_t buffer_size;
    sox_format_t *output = sox_open_memstream_write(
        &buffer, &buffer_size, &dst_signal, &dst_encoding, file_type, nullptr);
    if (output == nullptr) {
      throw std::runtime_error("Error opening output output memstream");
    }

    sox_format_t *input =
        sox_open_mem_read((void *)src.data(), src.size() * sizeof(sox_sample_t),
                          &src_signal, &src_encoding, file_type);

    if (input == nullptr) {
      throw std::runtime_error("Error opening output input memstream");
    }
    apply(input, output);
    sox_close(input);
    sox_close(output);

    auto output_length =
        dst_signal.length == 0 ? output->olength : dst_signal.length;

    sox_format_t *read_out = sox_open_mem_read(buffer, buffer_size, &dst_signal,
                                               &dst_encoding, file_type);
    // NB: it might happen that the output is 1 frame smaller than output_length
    // here we implicitly pad it with zeros
    dst.resize(output_length);

    const int64_t samples_read = sox_read(read_out, dst.data(), output_length);

    sox_close(read_out);
    free(buffer);
  }

  EffectChain() { init_sox_once(); }

  EffectChain &add_effect(const std::string &effect_name,
                          const std::vector<std::string> &effect_params) {
    if (SUPPORTED_EFFECTS.find(effect_name) == SUPPORTED_EFFECTS.end()) {
      std::ostringstream message;
      message << "Unsupported effect name: " << effect_name;
      throw std::runtime_error(message.str());
    }

    effects.emplace_back(
        sox_create_effect(sox_find_effect(effect_name.c_str())));

    if (effects.back() == nullptr) {
      std::ostringstream message;
      message << "Cannot create effect: " << effect_name;
      throw std::runtime_error(message.str());
    };

    effects.back()->global_info->global_info->verbosity = 1;

    auto num_opts = effect_params.size();
    if (num_opts == 0) {
      sox_effect_options(effects.back().get(), 0, nullptr);
    } else {
      char *sox_args[num_opts];
      for (auto i = 0; i < num_opts; ++i) {
        sox_args[i] = (char *)effect_params[i].c_str();
      }
      if (sox_effect_options(effects.back().get(), num_opts, sox_args) !=
          SOX_SUCCESS) {
        std::ostringstream message;
        message << "Cannot setup parameters for the effect: " << effect_name;
        throw std::runtime_error(message.str());
      }
    }
    return *this;
  }
};

} // namespace cpp_sox
