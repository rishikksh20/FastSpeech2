import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from dataset.texts import phonemes_to_sequence
import hparams as hp
import numpy as np
from dataset.texts import text_to_sequence
from utils.util import pad_list, str_to_int_list, remove_outlier

def get_tts_dataset(path, batch_size, valid=False) :

    if valid:
        file_ = hp.valid_filelist
        pin_mem = False
        num_workers = 0
    else:
        file_ = hp.train_filelist
        pin_mem = True
        num_workers = 4
    train_dataset = TTSDataset(path, file_)


    train_set = DataLoader(train_dataset,
                           collate_fn=collate_tts,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=True,
                           pin_memory=pin_mem)
    return train_set


class TTSDataset(Dataset):
    def __init__(self, path, file_) :
        self.path = path
        #self.f0_mean = np.load(f'{self.path}f0_mean.npy')
        #self.f0_std = np.load(f'{self.path}f0_std.npy')
        #self.e_mean = np.load(f'{self.path}e_mean.npy')
        #self.e_std = np.load(f'{self.path}e_std.npy')
        with open('{}'.format(file_), encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]

    def __getitem__(self, index):
        id = self._metadata[index][4].split(".")[0]
        x_ = self._metadata[index][3].split()
        if hp.use_phonemes:
            x = phonemes_to_sequence(x_)
        else:
            x = text_to_sequence(x_, hp.tts_cleaner_names)
        mel = np.load(f'{self.path}mels/{id}.npy')
        durations = str_to_int_list(self._metadata[index][2])
        e = remove_outlier(np.load(f'{self.path}energy/{id}.npy')) #self._norm_mean_std(np.load(f'{self.path}energy/{id}.npy'), self.e_mean, self.e_std, True)
        p = remove_outlier(np.load(f'{self.path}pitch/{id}.npy')) #self._norm_mean_std(np.load(f'{self.path}pitch/{id}.npy'), self.f0_mean, self.f0_std, True)
        mel_len = mel.shape[1]
        durations = durations[:len(x)]
        durations[-1] = durations[-1] + (mel.shape[1] - sum(durations))
        assert mel.shape[1] == sum(durations)
        return np.array(x), mel.T, id, mel_len, np.array(durations), e, p # Mel [T, num_mel]

    def __len__(self):
        return len(self._metadata)

    def _norm_mean_std(self, x, mean, std, is_remove_outlier = False):
        if is_remove_outlier:
            x = remove_outlier(x)
        zero_idxs = np.where(x == 0.0)[0]
        x = (x - mean) / std
        x[zero_idxs] = 0.0
        return x


def pad1d(x, max_len) :
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len) :
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')

def collate_tts(batch):

    ilens = torch.from_numpy(np.array([x[0].shape[0] for x in batch])).long()
    olens = torch.from_numpy(np.array([y[1].shape[0] for y in batch])).long()
    ids = [x[2] for x in batch]

    # perform padding and conversion to tensor
    inputs = pad_list([torch.from_numpy(x[0]).long() for x in batch], 0)
    mels = pad_list([torch.from_numpy(y[1]).float() for y in batch], 0)

    durations = pad_list([torch.from_numpy(x[4]).long() for x in batch], 0)
    energys = pad_list([torch.from_numpy(y[5]).float() for y in batch], 0)
    pitches = pad_list([torch.from_numpy(y[6]).float() for y in batch], 0)

    # make labels for stop prediction
    labels = mels.new_zeros(mels.size(0), mels.size(1))
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1.0

    # scale spectrograms to -4 <--> 4
    # mels = (mels * 8.) - 4

    return inputs, ilens, mels, labels, olens, ids, durations, energys, pitches

class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO : Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx) :
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)
