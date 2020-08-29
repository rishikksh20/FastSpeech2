from utils.util import get_files
import hparams as hp




def ljspeech(path, data_dir) :

    csv_file = get_files(path, extension='.csv')

    assert len(csv_file) == 1

    wavs = []
    #texts = []
    #encode = []

    with open(csv_file[0], encoding='utf-8') as f_ :
        # if 'phoneme_cleaners' in hp.tts_cleaner_names:
        #     print("Cleaner : {} Language Code : {}\n".format(hp.tts_cleaner_names[0],hp.phoneme_language))
        #     for line in f :
        #         split = line.split('|')
        #         text_dict[split[0]] = text2phone(split[-1].strip(),hp.phoneme_language)
        # else:
        print("Cleaner : {} \n".format(hp.tts_cleaner_names))
        for line in f_ :
            sub = {}
            split = line.split('|')
            t = split[-1].strip().upper()
            # t = t.replace('"', '')
            # t = t.replace('-', ' ')
            # t = t.replace(';','')
            # t = t.replace('(', '')
            # t = t.replace(')', '')
            # t = t.replace(':', '')
            # t = re.sub('[^A-Za-z0-9.!?,\' ]+', '', t)
            if len(t)>0:
                wavs.append(split[0].strip())
                #texts.append(t)
                #encode.append(text_to_sequence(t, hp.tts_cleaner_names))
    # with open(os.path.join(data_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    #     for w, t, e in zip(wavs, texts, encode):
    #         f.write('{}|{}|{}'.format(w,e,t) + '\n')


    return wavs #, texts, encode


if __name__ == "__main__":
    ljspeech('metadata.csv', ['english_cleaners'])
