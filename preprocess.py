from utils.display import *
from dataset.audio_processing import *
import hparams as hp
from multiprocessing import Pool, cpu_count
import os
import pickle
import argparse
from dataset.ljspeech import ljspeech
from utils.files import get_files


parser = argparse.ArgumentParser(description='Preprocessing for FastSpeech')
parser.add_argument('--path', '-p', default='./data/', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--out_path', '-o', default='./data/', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', default='.wav', help='file extension to search for in dataset folder')
args = parser.parse_args()

extension = args.extension
path = args.path




def convert_file(path) :
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    mel = melspectrogram(y)
    #mel = logmelspectrogram(y, hp.sample_rate, hp.n_mels, hp.n_fft, hp.hop_length)
    # Output [T, num_mel]

    e = energy(y)  # [T, ] T = Number of frames
    p = pitch(y)  # [T, ] T = Number of frames
    p = p[:mel.shape[1]] # Pitchs have some extra silence frame

    return  mel.astype(np.float32), e.astype(np.float32), p.astype(np.float32)


def process_wav(wav) :
    mel_path = os.path.join(args.out_path, 'mels')
    energy_path = os.path.join(args.out_path, 'energy')
    pitch_path = os.path.join(args.out_path, 'pitch')
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(energy_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)
    id = wav
    m, e, p = convert_file('{}/wavs/{}.wav'.format(path,wav))
    np.save('{}/{}.npy'.format(mel_path,id), m, allow_pickle=False)
    np.save('{}/{}.npy'.format(energy_path, id), e, allow_pickle=False)
    np.save('{}/{}.npy'.format(pitch_path, id), p, allow_pickle=False)
    return id, p.shape[0]


#wav_files = get_files(path, extension)
if __name__ == '__main__':
    os.makedirs(args.out_path, exist_ok=True)
    wav_files = ljspeech(path, args.out_path)
    print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

    if len(wav_files) == 0 :

        print('Please point wav_path in hparams.py to your dataset,')
        print('or use the --path option.\n')

    else :

        #with open(f'{args.out_path}text_dict.pkl', 'wb') as f:
        #    pickle.dump(text_dict, f)

        simple_table([('Sample Rate', hp.sample_rate),
                      ('Bit Depth', hp.bits),
                      ('Mu Law', hp.mu_law),
                      ('Hop Length', hp.hop_length),
                      ('CPU Count', cpu_count())])

        pool = Pool(processes= cpu_count())
        dataset = []

        for i, (id, length) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
            dataset += [(id, length)]
            bar = progbar(i, len(wav_files))
            message = f'{bar} {i}/{len(wav_files)} '
            stream(message)

        with open(f'{args.out_path}dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)



        print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
