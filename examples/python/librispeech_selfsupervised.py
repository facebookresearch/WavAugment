# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import torch

import torchaudio
from torchaudio.datasets.librispeech import FOLDER_IN_ARCHIVE, URL
from torchaudio.datasets.librispeech import LIBRISPEECH as Librispeech
from functools import lru_cache
import augment
import numpy as np
import os
import time
import argparse

"""
In this example, we implement a simple Librispeech-based dataset for self-supervised learning
with data augmentations implemented via WavAugment.
"""


# cache all files lengths in-mem to reduce disk IO
@lru_cache(maxsize=None)
def get_file_length(filepath):
    """
    Returns the length of the sequence in the file specified by `filepath`
    """
    signal_info, encoding_info = torchaudio.info(filepath)
    return signal_info.length


class LibrispeechSelfSupervised(Librispeech):
    """
    Extends the standard Librispeech dataset to a self-supervised use:
        * hides speaker and text labels
        * loads a sequences of speech of a predefined length, randomly shifted within a file
        * return two copies of this sequence, called `past` and `future`
        * those two sequences are independently augmented
    """

    def __init__(self, root, sequence_length, augment_past=None, augment_future=None, url=URL, folder_in_archive=FOLDER_IN_ARCHIVE, download=False):
        """
        root: where the dataset is stored
        sequence_length: expected length of the sequence
        augment_past: a Callable that applies data augmentation on `past` sequences
        augment_future: a Callable that applies data augmentation on `future` sequences
        """
        super().__init__(root, url, folder_in_archive, download)
        self.sequence_length = sequence_length
        self.augment_past = augment_past
        self.augment_future = augment_future

    def __getitem__(self, n):
        fileid = self._walker[n]
        waveform = self.load_librispeech_item(fileid)
        past, future = waveform, waveform

        if self.augment_past:
            past = self.augment_past(past)
        if self.augment_future:
            future = self.augment_future(future)

        return past, future

    def load_librispeech_item(self, fileid):
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_audio = fileid + self._ext_audio
        file_audio = os.path.join(
            self._path, speaker_id, chapter_id, file_audio)

        # Get the sequence length
        length = get_file_length(file_audio)

        assert length >= self.sequence_length, \
            f'Sequence {file_audio} is too short for the required length {self.sequence_length}'
        # Generate a random offset within the file
        offset = np.random.randint(0, length - self.sequence_length)

        # Load a randomly shifted piece of audio
        waveform, sample_rate = torchaudio.load(
            file_audio, offset=offset, num_frames=self.sequence_length)
        assert waveform.size(1) == self.sequence_length
        return waveform


class ChainRunner:
    """
    Takes an instance of augment.EffectChain and applies it on pytorch tensors.
    """

    def __init__(self, chain):
        self.chain = chain

    def __call__(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        src_info = {'channels': x.size(0),  # number of channels
                    'length': x.size(1),   # length of the sequence
                    'precision': 32,       # precision (16, 32 bits)
                    'rate': 16000.0,       # sampling rate
                    'bits_per_sample': 32}  # size of the sample

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 32,
                       'rate': 16000.0,
                       'bits_per_sample': 32}

        y, sampling_rate = self.chain.apply(
            x, src_info=src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        return y


# Generate a random shift applied to the speaker's pitch
def random_pitch_shift():
    return np.random.randint(-300, 300)

# Generate a random size of the room
def random_room_size():
    return np.random.randint(0, 100)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data', help='Where Librispeech is placed') 
    parser.add_argument('--download', action='store_true', help='Whether the dataset can be downloaded automatically if not found')
    parser.add_argument('--subset', choices=["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100",
            "train-clean-360", "train-other-500"], default='dev-clean', help='Librispeech subset to use')
    parser.add_argument('--sequence_length_seconds', type=int, default=1, help='Sample sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--n_workers', type=int, default=8, help="Number of parallel workers to read/preprocess data")
    parser.add_argument('--n_epochs', type=int, default=3, help="Number of epochs to run")
    parser.add_argument('--dump', action="store_true", help="Dump examples of (non)augmented sequences."
                                    "They would be saved in 'original.wav' and 'augmented.wav'")


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    effect_chain_past = augment.EffectChain()
    # The pitch effect changes the sampling ratio; we have to compensate for that.
    # Here, we specify 'quick' options on both pitch and rate effects, to speed up things
    effect_chain_past.pitch("-q", random_pitch_shift).rate("-q", 16_000)
    # Next effect we add is `reverb`; it adds makes the signal to have two channels,
    # which we combine into 1 by running `channels` w/o parameters
    effect_chain_past.reverb(50, 50, random_room_size).channels()
    # Futher, we add an effect that randomly drops one 50ms subsequence
    effect_chain_past.time_dropout(max_seconds=50 / 1000)

    effect_chain_past_runner = ChainRunner(effect_chain_past)

    # the second, `future` copy would be non-augmented
    effect_chain_future = None
    effect_chain_future_runner = None

    dataset = LibrispeechSelfSupervised(
        root=args.data,
        augment_past=effect_chain_past_runner,
        augment_future=effect_chain_future_runner,
        # In Librispeech, sampling rate is 16000
        sequence_length=args.sequence_length_seconds* 16_000,
        url=args.subset,
        download=args.download)

    if args.dump:
        augmented, original = dataset[0]
        torchaudio.save('augmented.wav', augmented, 16_000)
        torchaudio.save('original.wav', original, 16_000)
        print('Saved examples of augmented and non-augmented sequences to augmented.wav and original.wav')

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers
    )

    for epoch in range(args.n_epochs):
        start = time.time()
        for batch in dataloader:
            pass
        print(f'Finished epoch {epoch} in {time.time() - start}')
