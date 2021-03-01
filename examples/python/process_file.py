# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchaudio
import augment
import argparse

from dataclasses import dataclass

class RandomPitchShift:
    def __init__(self, shift_max=300):
        self.shift_max = shift_max

    def __call__(self):
        return np.random.randint(-self.shift_max, self.shift_max)

class RandomClipFactor:
    def __init__(self, factor_min=0.0, factor_max=1.0):
        self.factor_min = factor_min
        self.factor_max = factor_max
    def __call__(self):
        return np.random.triangular(self.factor_min, self.factor_max, self.factor_max)

@dataclass
class RandomReverb:
    reverberance_min: int = 50
    reverberance_max: int = 50
    damping_min: int = 50
    damping_max: int = 50
    room_scale_min: int = 0
    room_scale_max: int = 100

    def __call__(self):
        reverberance = np.random.randint(self.reverberance_min, self.reverberance_max + 1)
        damping = np.random.randint(self.damping_min, self.damping_max + 1)
        room_scale = np.random.randint(self.room_scale_min, self.room_scale_max + 1)

        return [reverberance, damping, room_scale]

class SpecAugmentBand:
    def __init__(self, sampling_rate, scaler):
        self.sampling_rate = sampling_rate
        self.scaler = scaler

    @staticmethod
    def freq2mel(f):
        return 2595. * np.log10(1 + f / 700)

    @staticmethod
    def mel2freq(m):
        return ((10.**(m / 2595.) - 1) * 700)

    def __call__(self):
        F = 27.0 * self.scaler
        melfmax = freq2mel(self.sample_rate / 2)
        meldf = np.random.uniform(0, melfmax * F / 256.)
        melf0 = np.random.uniform(0, melfmax - meldf)
        low = mel2freq(melf0)
        high = mel2freq(melf0 + meldf)
        return f'{high}-{low}'


def augmentation_factory(description, sampling_rate, args):
    chain = augment.EffectChain()
    description = description.split(',')

    for effect in description:
        if effect == 'bandreject':
            chain = chain.sinc('-a', '120', SpecAugmentBand(sampling_rate, args.band_scaler))
        elif effect == 'pitch':
            pitch_randomizer = RandomPitchShift(args.pitch_shift_max)
            if args.pitch_quick:
                chain = chain.pitch('-q', pitch_randomizer).rate('-q', sampling_rate)
            else:
                chain = chain.pitch(pitch_randomizer).rate(sampling_rate)
        elif effect == 'reverb':
            randomized_params = RandomReverb(args.reverberance_min, args.reverberance_max, 
                                args.damping_min, args.damping_max, args.room_scale_min, args.room_scale_max)
            chain = chain.reverb(randomized_params).channels()
        elif effect == 'time_drop':
            chain = chain.time_dropout(max_seconds=args.t_ms / 1000.0)
        elif effect == 'clip':
            chain = chain.clip(RandomClipFactor(args.clip_min, args.clip_max))
        elif effect == 'none':
            pass
        else:
            raise RuntimeError(f'Unknown augmentation type {effect}')
    return chain


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='File to procecss')
    parser.add_argument('--output_file', type=str, help='Output file')

    parser.add_argument('--chain', type=str, help='Comma-separated list of effects to apply, e.g. "pitch,dropout"',
        default='none')

    parser.add_argument('--t_ms', type=int, help='Size of a time dropout sequence', default=50)

    parser.add_argument('--pitch_shift_max', type=int, help='Amplitude of a pitch shift; measured in 1/100th of a tone', default=300)
    parser.add_argument('--pitch_quick', action='store_true', help='Speech up the pitch effect at some quality cost')
    
    parser.add_argument('--room_scale_min', type=int, help='Minimal room size used in randomized reverb (0..100)', default=0)
    parser.add_argument('--room_scale_max', type=int, help='Maximal room size used in randomized reverb (0..100)', default=100)
    parser.add_argument('--reverberance_min', type=int, help='Minimal reverberance used in randomized reverb (0..100)', default=50)
    parser.add_argument('--reverberance_max', type=int, help='Maximal reverberance used in randomized reverb (0..100)', default=50)
    parser.add_argument('--damping_min', type=int, help='Minimal damping used in randomized reverb (0..100)', default=50)
    parser.add_argument('--damping_max', type=int, help='Maximal damping used in randomized reverb (0..100)', default=50)
    parser.add_argument('--clip_min', type=float, help='Minimal clip factor (0.0..1.0)', default=0.5)
    parser.add_argument('--clip_max', type=float, help='Maximal clip factor (0.0..1.0)', default=1.0)

    args = parser.parse_args()
    args.chain = args.chain.lower()

    if not args.input_file or not args.output_file:
        raise RuntimeError('You need to specify "--input_file" and "--output_file"')

    if not (0 <= args.room_scale_min <= args.room_scale_max <= 100):
        raise RuntimeError('It should be that 0 <= room_scale_min <= room_scale_max <= 100')

    if not (0 <= args.reverberance_min <= args.reverberance_max <= 100):
        raise RuntimeError('It should be that 0 <= reverberance_min <= reverberance_max <= 100')

    if not (0 <= args.damping_min <= args.damping_max <= 100):
        raise RuntimeError('It should be that 0 <= damping_min <= damping_max <= 100')

    if not (0.0 <= args.clip_min <= args.clip_max <= 1.0):
        raise RuntimeError('It should be that 0 <= clip_min <= clip_max <= 1.0')

    return args

if __name__ == '__main__':
    args = get_args()

    x, sampling_rate = torchaudio.load(args.input_file)
    augmentation_chain = augmentation_factory(args.chain, sampling_rate, args)

    y = augmentation_chain.apply(x, 
            src_info=dict(rate=sampling_rate, length=x.size(1), channels=x.size(0)),
            target_info=dict(rate=sampling_rate, length=0)
    )

    torchaudio.save(args.output_file, y, sampling_rate)

