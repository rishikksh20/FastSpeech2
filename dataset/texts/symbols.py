""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from dataset.texts import cmudict

_pad = "_"
_eos = "~"
_bos = "^"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "abcdefghijklmnopqrstuvwxyz"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + [_eos]

# For Phonemes

PAD = "#"
EOS = "~"
PHONEME_CODES = "AA1 AE0 AE1 AH0 AH1 AO0 AO1 AW0 AW1 AY0 AY1 B CH D DH EH0 EH1 EU0 EU1 EY0 EY1 F G HH IH0 IH1 IY0 IY1 JH K L M N NG OW0 OW1 OY0 OY1 P R S SH T TH UH0 UH1 UW0 UW1 V W Y Z ZH pau".split()
_PHONEME_SEP = " "

phonemes_symbols = [PAD, EOS] + PHONEME_CODES  # PAD should be first to have zero id
