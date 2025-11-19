import scipy
from pathlib import Path
import scipy.io
import gc
import h5py
from itertools import product

from torch.utils.data import Dataset
from src.config import Config
import logging
logger = logging.getLogger(__name__)
import dask
import dask.array as da
import os
import numpy as np
import torch
from src.utils.phoneme_utils import text_to_phonemes, phoneme_to_id, clean_text, PHONE_DEF_SIL


class SimpleTokenizer:
    def __init__(self):
        # Create a mapping of characters to integers including apostrophe
        self.char_to_index = {chr(i): i - 96 for i in range(97, 123)}  # 'a' to 'z'
        self.char_to_index["'"] = 27  # Adding apostrophe

    def tokenize(self, text):
        return [self.char_to_index.get(char, 0) for char in text.lower() if char in self.char_to_index]

    def encode(self, text):
        # Tokenize the text and calculate lengths
        token_ids = self.tokenize(text)
        return token_ids, len(token_ids)

def get_dataset(batch_size, n_batches, get_phonemes=False):
    h5path = str(Config.DIRS.H5_DIR)
    if os.path.exists(h5path):
        dataset = HDF5Dataset(h5path, batch_size=batch_size, n_batches=n_batches, get_phonemes=get_phonemes)
    else:
        # make h5 file using the MatDataset and then use DaskHDF5Dataset
        dataset = MatDataset(Config.DIRS.MAT.TRAIN)
        print("Created MatDataset for Further Use...")
        f = h5py.File(h5path, 'w')
        print("Created hdf5 file for HDF5Dataset...")
        n_sentences = 0
        f.create_group("globals")
        try:
            for i, sentence in enumerate(dataset):
                n_sentences += 1
                print(f"writing sentence {i}")
                signals = sentence['signals']
                text = sentence['text']
                tokenizer = tokenizer(text)
                phonemes = sentence['phonemes']
                tokens = sentence['phonemes_ids']
                date = sentence['date']
                f.create_group(f'sentence_{i}')
                f[f'sentence_{i}'].create_dataset('signals', data=signals)
                f[f'sentence_{i}'].attrs['text'] = text
                f[f'sentence_{i}'].attrs['phonemes'] = phonemes
                f[f'sentence_{i}'].attrs['phonemes_ids'] = tokens
                f[f'sentence_{i}'].attrs['date'] = date
                f[f'sentence_{i}'].attrs['tokenizer'] = tokenizer
            print(n_sentences)
            f['globals'].attrs['n_sentences'] = n_sentences
            f.close()
            print("Closing the hdf5 file...")
            del dataset
            gc.collect()
            tokenizer = SimpleTokenizer()
            dataset = HDF5Dataset(h5path,batch_size=batch_size,
                                  tokenizer=tokenizer, n_batches=n_batches, get_phonemes=get_phonemes)
            for data in dataset:
                print(data['text'])
                print(data['token_ids'])
                print(data['token_lengths'])
        except Exception as e:
            f.close()
            os.remove(h5path)
            raise e
    return dataset


class HDF5Dataset(Dataset):
    def __init__(self, path, batch_size=1, n_batches=1, tokenizer=None, get_phonemes=False, time_warping=True,
                 window_size=50, alpha=None):
        self.tokenizer = tokenizer  # Pass an instance of SimpleTokenizer when initializing the dataset
        self.get_phonemes = get_phonemes
        self.path = path
        self.batch_size = batch_size
        self.time_warping = time_warping
        self.window_size = window_size
        self.alpha = alpha
        self.n_batches = n_batches
        if not self.batch_size in [1, 2, 4, 5, 10, 20]:
            raise ValueError(f'Batch size must be one of [1, 2, 5, 10, 20]')
        f = h5py.File(self.path, 'r')
        self.n_sentences = int(f['globals'].attrs['n_sentences'])
        if n_batches == 'all':
            self.n_batches = self.n_sentences // self.batch_size
        elif n_batches > self.n_sentences // self.batch_size:
            raise ValueError(f'n_batches must be less than {self.n_sentences // self.batch_size}')
        else:
            self.n_batches = n_batches
        f.close()

    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as f:
            sentence_group = f[f'sentence_{idx}']
            text = str(sentence_group.attrs['text'])
            token_ids, token_lengths = self.tokenizer.encode(text)
            phonemes = list(sentence_group.attrs['phonemes'])
            phoneme_ids = torch.tensor(list(map(int, sentence_group.attrs['phonemes_ids'])))
            signals = torch.from_numpy(np.array(sentence_group['signals'])).float()

        return {
            'signals': signals,
            'text': text,
            'phonemes': phonemes,
            'phoneme_ids': phoneme_ids,
            'token_ids': torch.tensor(token_ids),
            'token_lengths': token_lengths
        }

    def __len__(self):
        return self.n_sentences

    def __iter__(self): # RANDOM BATCH ITERATION
        index = np.arange(self.n_batches)
        permutation = np.random.permutation(index)
        for batch_n in permutation:
            yield self._get_batch(int(batch_n))

    def __getitem__(self, idx):
        # Load and return a batch of data
        return self._get_batch(idx)

    def __len__(self):
        return self.n_batches

    def _get_batch(self, batch_n):
        slice_start = batch_n * self.batch_size
        slice_stop = (batch_n + 1) * self.batch_size
        if slice_stop > self.n_sentences:
            slice_stop = self.n_sentences
        slice_obj = slice(slice_start, slice_stop)
        sentences = self._get_slice(slice_obj)
        batch = {}
        for key in sentences[0].keys():
            batch[key] = list(map(lambda x: x[key], sentences))
            print(f"Key: {key}, Length: {len(batch[key])}")
        if slice_start > 0:
            prev_sentence = self._get_sentence(slice_start - 1)
            prev_signal = prev_sentence['signals']
        else:
            prev_signal = None
        aug_signals = self._augment_batch(batch['signals'], prev_signal)
        if self.get_phonemes:
            batch['signals'], batch['signal_lengths'] = self._pad_and_stack_signals(aug_signals)
            if self.get_phonemes:
                tokens= batch['phoneme_ids']
                padded_phoneme_ids, phoneme_lengths = self._pad_and_stack_phoneme_ids(tokens)
                batch['phoneme_ids'] = padded_phoneme_ids
                batch['phoneme_lengths'] = phoneme_lengths
        else:
            batch['signals'] = aug_signals[0]
            #if self.get_phonemes:
            #    # signals.shape (1, T, 4, 3, 8, 8)
            #    batch['signal_lengths'] = torch.Tensor(int(batch['signals'].shape[1]))
            #    batch['phoneme_ids'] = batch['phoneme_ids'][0].unsqueeze(0).long()
            #    batch['phoneme_lengths'] = torch.Tensor(int(batch['phoneme_ids'].shape[1])).long()
        return batch

    def _pad_and_stack_signals(self, signals):
        """Zero Pads and stacks the signals in the batch.
        Returns the accompanying tensor of lengths for each signal."""
        max_len = Config["MODEL_PARAMETERS"]["FIXED_NEURAL_SEQUENCE_LENGTH"]
        signal_lengths = torch.tensor([signal.shape[1] for signal in signals])
        padded_signals = []
        for signal in signals:
            padded_signal = torch.zeros((1, max_len, *signal.shape[2:]))
            padded_signal[:,0:signal.shape[1],:,:,:] = signal
            padded_signals.append(padded_signal)
        return torch.cat(padded_signals, dim=0), signal_lengths

    def _pad_and_stack_phoneme_ids(self, phoneme_ids_list):
        """Zero Pads and stacks the phoneme id labels in the batch.
        Returns the accompanying tensor of lengths for each."""
        print(f"Phoneme IDs List: {len(phoneme_ids_list)}")
        max_len = max([len(p) for p in phoneme_ids_list])
        phoneme_lengths = torch.tensor([len(p) for p in phoneme_ids_list]).long()
        padded_phoneme_ids = []
        for tokens in phoneme_ids_list:
            padded_phoneme_id = torch.zeros((max_len))
            padded_phoneme_id[0:len(tokens)] = tokens
            padded_phoneme_ids.append(padded_phoneme_id)
            tokens= torch.stack(padded_phoneme_ids, dim=0).long()
            print(f"Phoneme IDs: {tokens.shape}")
        return tokens, phoneme_lengths



    def _augment_batch(self, signals, prev_signal):
        """Applies EMA smoothing and rolling
        normalization to the batch of signals."""
        augmented_signals = []
        for signal in signals:
            prev_signal = signal.clone()
            aug_signal = self._time_warp(signal)
            # check for nans
            #print("Time Warp: ", torch.isnan(aug_signal).any())
            aug_signal = self._add_noise(aug_signal)
            #print("Noise: ", torch.isnan(aug_signal).any())
            aug_signal = self._smooth(signal)
            #print("Smooth: ", torch.isnan(aug_signal).any())
            aug_signal = self._normalize(signal, prev_signal)
            #print("Normalize: ", torch.isnan(aug_signal).any())
            augmented_signals.append(aug_signal)
        return augmented_signals

    def _time_warp(self, tensor, warp_factor_range=(0.8, 1.25)):
        """
        Apply time warping to a one-sentence tensor of shape (1, T, A, C, H, W).

        Args:
        - tensor (torch.Tensor): A tensor of shape (1, T, A, C, H, W).
        - warp_factor_range (float, float): A tuple indicating the min and max range for the warp factor.

        Returns:
        - torch.Tensor: A time-warped tensor.
        """
        # tensor shape
        time_steps = tensor.shape[1]
        other_dims = tensor.shape[2:]
        warp_factor = torch.empty(1).uniform_(*warp_factor_range).to(tensor.device)

        # Generate warped time indices
        warped_time_steps = (torch.arange(time_steps).float() * warp_factor).long()
        warped_time_steps = torch.clamp(warped_time_steps, 0, time_steps - 1)

        # Apply time warping
        warped_tensor = torch.gather(tensor, 1, warped_time_steps.view(1, -1, 1, 1, 1, 1).expand(-1, -1, *other_dims))

        return warped_tensor

    def _smooth(self, signal):
        """
        Applies Gaussian smoothing to each frame.
        Then applies Exponential Weighted Moving Average
        smoothing to each channel."""
        signal = self._smooth_frames(signal)
        signal = self._smooth_channels(signal)
        return signal

    def _smooth_frames(self, signal):
        """Applies gaussian smoothing to frames."""
        # signal shape (1, T, 4, 3, 8, 8)
        grid = product(*map(lambda x: range(x), list(signal.shape[1:4])))
        for t, a, c  in grid:
            signal[0,t,a,c,:,:] = self._smooth_frame(signal[0,t,a,c,:,:])
        return signal

    def _smooth_frame(self, frame):
        """Applies gaussian smoothing to a single 8x8 frame."""
        from scipy.ndimage import gaussian_filter
        return torch.tensor(gaussian_filter(frame, sigma=1, mode='nearest'))

    def _smooth_channels(self, signal):
        """Applies Exponential Weighted Moving Average smoothing to each channel."""
        grid = product(*map(lambda x: range(x), list(signal.shape[2:])))
        for (a, c, y, x) in grid:
            signal[0,:,a,c,y,x] = self._smooth_channel(signal[0,:,a,c,y,x])
        return signal

    def _smooth_channel(self, channel):
        from scipy.signal import lfiltic, lfilter
        if self.alpha is None:
            self.alpha = 2 /(self.window_size + 1)
        b = [self.alpha]
        a = [1, self.alpha-1]
        zi = lfiltic(b, a, channel[0:1], [0])
        return torch.tensor(lfilter(b, a, channel, zi=zi)[0])

    def _add_noise(self, signal):
        """Adds Gaussian noise to the sparse signals."""
        noise_shape = list(signal.shape)
        noise_shape[3] = noise_shape[3] - 1
        noise_shape = tuple(noise_shape)
        signal[:,:,:,1:,:,:] += torch.randn(noise_shape).to(signal.device)
        return signal

    def _normalize(self, signal, prev_signal):
        """Performs rolling normalization on the signal.
        Computes the cumulative mean and standard deviation
        updating for each new frame at the channel-wise level.
        """
        grid = product(*map(lambda x: range(x), list(signal.shape[2:])))
        if prev_signal is None:
            for (a, c, y, x) in grid:
                signal[0,:,a,c,y,x] = self._normalize_channel(signal[0,:,a,c,y,x], None)
        else:
            for (a, c, y, x) in grid:
                signal[0,:,a,c,y,x] = self._normalize_channel(signal[0,:,a,c,y,x], prev_signal[0,:,a,c,y,x])
        return signal

    def _normalize_channel(self, channel, prev_channel):
        return (channel - channel.mean()) / channel.std()

    '''
    def _normalize_channel(self, channel, prev_channel):
        """Performs rolling normalization on a single channel."""
        # compute cumulative mean and std
        # update mean and std for each new frame
        # normalize each frame
        if prev_channel is not None:
            prev_mean = prev_channel.mean()
            prev_std = prev_channel.std()
        else:
            prev_mean = 0
            prev_std = 1

        running_sum = 0
        running_sum_squares = 0

        for i in range(channel.shape[0]):
            running_sum += channel[i]
            running_sum_squares += channel[i]**2
            if i < 50:
                p = i/50
                q = 1 - p
                mean = p * (running_sum/(i+1)) + q * prev_mean
                std = p * ((running_sum_squares / (i+1) - (running_sum / (i+1))**2)**0.5) + q * prev_std
            else:
                mean = running_sum / (i+1)
                std = (running_sum_squares / (i+1) - (running_sum / (i+1))**2)**0.5
            channel[i] = (channel[i] - mean) / std
        return channel
    '''

    def _get_slice(self, slice_obj):
        return [self._get_sentence(i) for i in range(slice_obj.start, slice_obj.stop)]

    def _get_sentence(self, idx):
        f = h5py.File(self.path, 'r')
        sentence_group = f[f'sentence_{idx}']

        # Other metadata can be read normally
        result = {
            'signals': torch.from_numpy(np.array(sentence_group['signals'])).float(),
            'date': str(sentence_group.attrs['date']),
        }
        if self.get_phonemes:
            result['text'] = str(sentence_group.attrs['text'])
            result['phonemes'] = list(sentence_group.attrs['phonemes'])
            tokens = list(map(int, sentence_group.attrs['phonemes_ids']))
            sot = int(Config["DATA"]["START_TOKEN"])
            eot = int(Config["DATA"]["END_TOKEN"])
            result['phoneme_ids'] = torch.tensor([sot] + tokens + [eot]).long()
        f.close()
        gc.collect()
        return result

class MatDataset(Dataset):
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.mat_file_paths = sorted(list(self.directory.glob('*.mat')))
        self.mat_files = []
        for path in self.mat_file_paths:
            try:
                mat_file = MatFile(path)
            except:
                print(f'Error loading {path}. Excluding from dataset.')
                continue
            self.mat_files.append(mat_file)
        self.n_days = len(self.mat_files)
        self.n_blocks = sum([f.n_blocks for f in self.mat_files])
        self.n_sentences = sum([f.n_sentences for f in self.mat_files])

    def __iter__(self):
        for mat_file in self.mat_files:
            for sentence in mat_file:
                yield sentence

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_sentence(idx)
        elif isinstance(idx, slice):
            return [self.get_sentence(i) for i in range(idx.start, idx.stop)]
        elif isinstance(idx, list):
            return [self.get_sentence(i) for i in idx]
        elif isinstance(idx, np.ndarray):
            return [self.get_sentence(i) for i in idx]
        elif isinstance(idx, torch.Tensor):
            return [self.get_sentence(i) for i in idx.numpy()]
        else:
            raise TypeError(f'Invalid argument type {type(idx)}')

    def get_sentence(self, idx):
        for f in self.mat_files:
            if idx < f.n_sentences:
                return f
            else:
                idx -= f.n_sentences
        raise IndexError(f'Index {idx} out of range')

class MatFile:
    def __init__(self, path: str):
        self.path = Path(path)
        self.mat = scipy.io.loadmat(str(path))
        self.date = self.path.stem.split('.')[1:]
        self.date = f'{self.date[0]}-{self.date[1]}-{self.date[2]}'
        self.sentence_text = self.mat['sentenceText']
        self.block_idx = self.mat['blockIdx']
        self.spike_pow = self.mat['spikePow']
        self.tx1 = self.mat['tx1']
        self.tx2 = self.mat['tx2']
        self.tx3 = self.mat['tx3']
        self.tx4 = self.mat['tx4']
        self.n_sentences = self.sentence_text.shape[0]
        self.n_blocks = len(np.unique(self.block_idx))
        self.block_id_map = {i: block_id for i, block_id in enumerate(np.unique(self.block_idx))}

    def __repr__(self):
        return f"MatFile({self.path})"

    def __str__(self):
        return f"MatFile({self.path})"

    def __len__(self):
        return self.n_sentences

    def __iter__(self):
        for i in range(self.n_sentences):
            yield self._get_sentence(i)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_sentence(idx)
        elif isinstance(idx, slice):
            return [self._get_sentence(i) for i in range(idx.start, idx.stop)]
        elif isinstance(idx, list):
            return [self._get_sentence(i) for i in idx]
        elif isinstance(idx, np.ndarray):
            return [self._get_sentence(i) for i in idx]
        else:
            raise TypeError(f'Invalid argument type {type(idx)}')

    def _get_sentence(self, idx:int):
        arrays = [self.spike_pow[0,idx], self.tx1[0,idx], self.tx4[0,idx]]
        area_6v_inferior = map_arrays(arrays, 'area_6v', 'inferior')
        area_6v_superior = map_arrays(arrays, 'area_6v', 'superior')
        area_44_inferior = map_arrays(arrays, 'area_44', 'inferior')
        area_44_superior = map_arrays(arrays, 'area_44', 'superior')
        signal = np.concatenate([area_6v_inferior, area_6v_superior, area_44_inferior, area_44_superior], axis=1)
        signal = torch.from_numpy(signal).float()
        # expand dims to (1, T, 4, CH, 8, 8)
        signal = signal.unsqueeze(0)
        text = self.sentence_text[idx]
        text = clean_text(text)
        phonemes = text_to_phonemes(text)
        tokens= [phoneme_to_id(p) for p in phonemes]
        return {
            'date': self.date,
            'text': text,
            'phonemes': phonemes,
            'phonemes_ids': tokens,
            'signals': signal
        }

def map_arrays(arrays, area, subarea):
    CH = len(arrays)
    T = arrays[0].shape[0]
    # Change to a Dask array if necessary
    new_array = np.zeros((T, 1, CH, 8, 8))
    for a, array in enumerate(arrays):
        channel_map = CHANNEL_MAPS[area][subarea]
        for i in range(8):
            for j in range(8):
                new_array[:,0,a,i,j] = array[:,channel_map[i][j]]
    return new_array


logging.basicConfig(filename='dask_dataset.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHANNEL_MAPS = {
    "area_44": {
        "superior": [
            [192, 193, 208, 216, 160, 165, 178, 185],
            [194, 195, 209, 217, 162, 167, 180, 184],
            [196, 197, 211, 218, 164, 170, 177, 189],
            [198, 199, 210, 219, 166, 174, 173, 187],
            [200, 201, 213, 220, 168, 176, 183, 186],
            [202, 203, 212, 221, 172, 175, 182, 191],
            [204, 205, 214, 223, 161, 169, 181, 188],
            [206, 207, 215, 222, 163, 171, 179, 190]
        ],
        "inferior": [
            [129, 144, 150, 158, 224, 232, 239, 255],
            [128, 142, 152, 145, 226, 233, 242, 241],
            [130, 135, 148, 149, 225, 234, 244, 243],
            [131, 138, 141, 151, 227, 235, 246, 245],
            [134, 140, 143, 153, 228, 236, 248, 247],
            [132, 146, 147, 155, 229, 237, 250, 249],
            [133, 137, 154, 157, 230, 238, 252, 251],
            [136, 139, 156, 159, 231, 240, 254, 253]
        ]
    },
    "area_6v": {
        "superior": [
            [62, 51, 43, 35, 94, 87, 79, 78],
            [60, 53, 41, 33, 95, 86, 77, 76],
            [63, 54, 47, 44, 93, 84, 75, 74],
            [58, 55, 48, 40, 92, 85, 73, 72],
            [59, 45, 46, 38, 91, 82, 71, 70],
            [61, 49, 42, 36, 90, 83, 69, 68],
            [56, 52, 39, 34, 89, 81, 67, 66],
            [57, 50, 37, 32, 88, 80, 65, 64]
        ],
        "inferior": [
            [125, 126, 112, 103, 31, 28, 11, 8],
            [123, 124, 110, 102, 29, 26, 9, 5],
            [121, 122, 109, 101, 27, 19, 18, 4],
            [119, 120, 108, 100, 25, 15, 12, 6],
            [117, 118, 107, 99, 23, 13, 10, 3],
            [115, 116, 106, 97, 21, 20, 7, 2],
            [113, 114, 105, 98, 17, 24, 14, 0],
            [127, 111, 104, 96, 30, 22, 16, 1]
        ]
    }
}


