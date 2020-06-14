import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from dataset.texts import phonemes_to_sequence
import hparams as hp
import numpy as np
from dataset.texts import text_to_sequence
from utils.util import pad_list, str_to_int_list

def get_tts_dataset(path, batch_size, valid=False) :

    with open(f'{path}dataset.pkl', 'rb') as f :
        dataset = pickle.load(f)

    dataset_ids = []
    mel_lengths = []
    print("Cleaner : {}".format(hp.tts_cleaner_names))
    for (id, len) in dataset :
        if len <= hp.tts_max_mel_len :
            dataset_ids += [id]
            mel_lengths += [len]

    # with open(f'{path}text_dict.pkl', 'rb') as f:
    #     text_dict = pickle.load(f)
    if valid:
        file_ = hp.valid_filelist
    else:
        file_ = hp.train_filelist
    train_dataset = TTSDataset(path, file_)

    # sampler = None
    #
    # if hp.tts_bin_lengths :
    #      sampler = BinnedLengthSampler(mel_lengths, batch_size, batch_size*3)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate_tts,
                           batch_size=batch_size,
                           #sampler=sampler,
                           num_workers=0,
                           shuffle=True)
                           #pin_memory=True)

    #longest = mel_lengths.index(max(mel_lengths))
    #attn_example = dataset_ids[longest]
    #print("Longest mels :",longest)
    #print("Attn exp :",attn_example)
    # print(attn_example)

    return train_set


class TTSDataset(Dataset):
    def __init__(self, file_) :
        #self.path = path
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
        e = np.load(f'{self.path}energy/{id}.npy')
        p = np.load(f'{self.path}pitch/{id}.npy')
        mel_len = mel.shape[0]
        return np.array(x), mel, id, mel_len, np.array(durations), e, p

    def __len__(self):
        return len(self._metadata)


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
