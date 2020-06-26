# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Callable, Union, Tuple

import torch
import numpy as np
import augment_cpp as _augment


def shutdown_sox() -> None:
    _augment.shutdown_sox()


# Arguments that we can pass to effects
SoxArg = Optional[List[Union[str, int, Callable]]]


class SoxEffect:
    def __init__(self, name: str, args: SoxArg):
        self.name = name
        self.args = args if args else []

    def instantiate(self):
        """
        >>> import random; random.seed(7)
        >>> effect = SoxEffect("pitch", [lambda: random.randint(-100, 100)])
        >>> effect.instantiate()
        ['pitch', ['-18']]
        >>> effect.instantiate()
        ['pitch', ['-62']]
        """
        instantiated_args = []
        for arg in self.args:
            if callable(arg):
                arg = arg()
                if isinstance(arg, list):
                    instantiated_args.extend([str(v) for v in arg])
                else:
                    instantiated_args.append(str(arg))
            else:
                instantiated_args.append(str(arg))

        return [self.name, instantiated_args]


class EffectChain:
    """
    >>> chain = EffectChain()
    >>> _ = chain.pitch("100").rate(16_000).dither()
    >>> len(chain)
    3
    >>> [e.name for e in chain._chain]
    ['pitch', 'rate', 'dither']
    """

    # libsox sample_t is an int between [-1 << 31, 1 << 31);
    # while torchaudio operates with [-1, 1]. Hence,
    # each time we pass something to/from libsox, we rescale
    _NORMALIZER: int = 1 << 31

    def __init__(self, in_place: bool = False):
        self._chain: List[Union[Callable, SoxEffect]] = []
        self.in_place: bool = in_place

    def _append_effect_to_chain(self, name: str, args: SoxArg = None):
        effect = SoxEffect(name, args)
        self._chain.append(effect)
        return self

    def clear(self):
        self._chain = []

    def __len__(self):
        return len(self._chain)

    @staticmethod
    def _apply_sox_effects(chain: List[SoxEffect],
                           input_tensor: torch.Tensor,
                           src_info: Dict,
                           target_info: Dict) -> Tuple[torch.Tensor, int]:
        instantiated_chain = [x.instantiate() for x in chain]
        sox_chain = _augment.PyEffectChain()
        for effect_name, effect_args in instantiated_chain:
            sox_chain.add_effect(effect_name, effect_args)

        input_tensor.mul_(EffectChain._NORMALIZER)

        out = input_tensor
        sr = sox_chain.apply_flow_effects(input_tensor,
                                          out,
                                          src_info,
                                          target_info)
        out.div_(EffectChain._NORMALIZER)
        return out, sr

    def apply(self,
              input_tensor: torch.Tensor,
              src_info: Dict[str, Union[int, float]],
              target_info: Dict[str, Union[int, float]]) -> torch.Tensor:
        """
        input_tensor (torch.Tensor): the input wave to be transformed;
            expected shape is (n_channels, length). If it has only
            one dimension, it is automatically expanded to have 1 channel.
        src_info (Dict): description of the input signal
        target_info (Dict): description of the output signal

        Fields that src_info and target_info can contain:
            * rate (mandatory for input),
            * length,
            * precision,
            * bits_per_sample.
        Those fields are only used by sox-based effects.

        Minimally, src_info must contain rate field (e.g. `{rate: 16_000}`).
        Both src_info and target_info can set `length`. If src_info specifies
        `length`, only first `length` samples are used; if target_info has `length`,
        further output is trimmed.

        It is might happen that sox will return wave shorter than `length` - in this
        case the output will be padded with zeroes.

        returns: torch.Tensor with transformed sound.
        """

        if not torch.is_tensor(input_tensor) or input_tensor.is_cuda:
            raise RuntimeError('Expected a CPU tensor')

        if not self.in_place:
            input_tensor = input_tensor.clone()
        if 'rate' not in src_info:
            raise RuntimeError("'rate' must be specified for the input")
        if len(input_tensor.size()) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if 'length' in src_info and src_info['length'] > input_tensor.size(1):
            raise RuntimeError("'length' is beyond the tensor length")

        src_info = dict(src_info)  # can be mutated in process
        sr = src_info['rate']

        if not self._chain:
            out = input_tensor
            return out

        sox_effects: List[SoxEffect] = []
        x = input_tensor

        # To minimize the number of sox calls, we stack consequent sox effects together in a single
        # sox-side chain. In contrast, we apply python effects immediately.

        for effect in self._chain:
            if callable(effect):
                if sox_effects:
                    x, sr = EffectChain._apply_sox_effects(
                        sox_effects, x, src_info, target_info)
                    src_info = dict(target_info)
                    assert src_info['rate'] == sr

                    sox_effects = []
                # effect should not mutate src_info or target_info, but
                # return new ones if changed
                x, src_info, target_info = effect(x, src_info, target_info)
            elif isinstance(effect, SoxEffect):
                sox_effects.append(effect)
            else:
                assert False

        if sox_effects:
            x, _ = EffectChain._apply_sox_effects(
                sox_effects, x, src_info, target_info)
        return x

    def time_dropout(self, max_frames: Optional[int] = None, max_seconds: Optional[float] = None):
        """
        >>> np.random.seed(1)
        >>> chain = EffectChain().time_dropout(max_seconds=0.1)
        >>> t = torch.ones([1, 16000])
        >>> x = chain.apply(t, {'rate': 16000}, {'rate': 16000})
        >>> x.min().item(), x.max().item()
        (0.0, 1.0)
        >>> (x == 0).sum().item()
        1061
        >>> (x[:, 235:1296] == 0).all().item()
        True
        >>> (x[:, :235] == 0).any().item()
        False
        >>> (x[:, 235 + 1061 + 1:] == 0).any().item()
        False
        """
        self._chain.append(TimeDropout(max_frames, max_seconds))

        return self

    def additive_noise(self, noise_generator: Callable, snr: float):
        """
        >>> signal = torch.zeros((1, 100)).uniform_()
        >>> noise_generator = lambda: torch.zeros((1, 100))
        >>> chain = EffectChain().additive_noise(noise_generator, snr=0)
        >>> x = chain.apply(signal, {'rate': 16000}, {'rate': 16000})
        >>> (x == signal.mul(0.5)).all().item()
        True
        """
        self._chain.append(AdditiveNoise(
            noise_generator=noise_generator, snr=snr))
        return self

    def clip(self, factor_generator: Callable):
        """
        >>> signal = torch.tensor([-10, -5, 0, 5, 10]).float()
        >>> factor_generator = lambda: 0.5
        >>> chain = EffectChain().clip(factor_generator)
        >>> x = chain.apply(signal, {'rate': 16000}, {})
        >>> x
        tensor([[-5., -5.,  0.,  5.,  5.]])
        """
        self._chain.append(ClipValue(factor_generator))
        return self


class TimeDropout:
    def __init__(self, max_frames: Optional[int] = None, max_seconds: Optional[float] = None):
        assert max_frames or max_seconds
        self.max_frames = max_frames
        self.max_seconds = max_seconds

    def __call__(self, x, src_info, dst_info):
        if self.max_frames is None:
            max_frames = int(src_info['rate'] * self.max_seconds)
        else:
            max_frames = self.max_frames

        length = np.random.randint(0, max_frames)

        start = np.random.randint(0, x.size(1) - length)
        end = start + length

        x[:, start:end, ...].zero_()
        return x, src_info, dst_info


class AdditiveNoise:
    def __init__(self, noise_generator: Callable, snr: float):
        self.noise_generator = noise_generator
        self.snr = snr

        r = np.exp(snr * np.log(10) / 10)
        self.coeff = r / (1.0 + r)

    def __call__(self, x, src_info, dst_info):
        noise_instance = self.noise_generator()
        assert noise_instance.numel() == x.numel(
        ), 'Noise and signal shapes are incompatible'

        noised = self.coeff * x + (1.0 - self.coeff) * noise_instance.view_as(x)
        return noised, src_info, dst_info


class ClipValue:
    def __init__(self, value_generator: Callable):
        self.value_generator = value_generator

    def __call__(self, x, src_info, dst_info):
        factor = self.value_generator()
        x_min, x_max = x.min(), x.max()

        x.clamp_(min=x_min * factor, max=x_max * factor)
        return x, src_info, dst_info


def create_method(name):
    return lambda s, *x: s._append_effect_to_chain(name, list(x)) # pylint: disable=protected-access


for _effect_name in _augment.get_effect_names():
    setattr(EffectChain, _effect_name, create_method(_effect_name))
