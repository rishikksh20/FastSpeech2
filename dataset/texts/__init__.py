""" from https://github.com/keithito/tacotron """
import re
from dataset.texts import cleaners
from dataset.texts.symbols import symbols, _eos, phonemes_symbols, PAD, EOS, _PHONEME_SEP
import hparams as hp
from dataset.texts.dict_ import symbols_

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

symbols_inv = {v: k for k, v in symbols_.items()}

def pad_with_eos_bos(_sequence):
    return _sequence + [_symbol_to_id[_eos]]



def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []
    if hp.eos:
        text = text + '~'
    try:
        sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
    except KeyError:
        print("text : ",text)
        exit(0)

    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in symbols_inv:
            s = symbols_inv[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [symbols_[s.upper()] for s in symbols]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


# For phonemes
_phoneme_to_id = {s: i for i, s in enumerate(phonemes_symbols)}
_id_to_phoneme = {i: s for i, s in enumerate(phonemes_symbols)}


def _should_keep_token(token, token_dict):
    return token in token_dict \
           and token != PAD and token != EOS \
           and token != _phoneme_to_id[PAD] \
           and token != _phoneme_to_id[EOS]

def phonemes_to_sequence(phonemes):
    string = phonemes.split(_PHONEME_SEP) if isinstance(phonemes, str) else phonemes
    string.append(EOS)
    sequence = [_phoneme_to_id[s] for s in string
                if _should_keep_token(s, _phoneme_to_id)]
    return sequence


def sequence_to_phonemes(sequence, use_eos=False):
    string = [_id_to_phoneme[idx] for idx in sequence
              if _should_keep_token(idx, _id_to_phoneme)]
    string = _PHONEME_SEP.join(string)
    if use_eos:
        string = string.replace(EOS, '')
    return string
