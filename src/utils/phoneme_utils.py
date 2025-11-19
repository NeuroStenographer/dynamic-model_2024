import unicodedata
import re
import string

from g2p_en import G2p

def convert_to_ascii(text):
    return [ord(char) for char in text]

def phoneme_to_id(p):
    return PHONE_DEF_SIL.index(p)

def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', str(text))
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation + 'â' + '\u00BF\u00A1')
    text = text.translate(translator)
    text = re.sub(r'\s\s+', ' ' , text).strip()
    text = re.sub(r'[^a-zA-Z\- \']', '', text)
    text = text.replace('--', '').lower()
    return text

def text_to_phonemes(text):
    """Convert text to phonemes using g2p_en.
    Args:
        text (str): input text
    Returns:
        phonemes (list): list of phonemes
    Notes:
    ------
    Uses G2p phonemizer from g2p_en package.
    From the g2p_en documentation:
        - Spells out arabic numbers and some currency symbols. (e.g. $200 -> two hundred dollars) (This is borrowed from Keith Ito’s code)
        - Attempts to retrieve the correct pronunciation for homographs based on their POS)
        - Looks up The CMU Pronouncing Dictionary for non-homographs.
        - For OOVs, we predict their pronunciations using our neural net model.
    """
    g2p=G2p()
    phonemes = []
    if len(text) == 0:
        phonemes = SIL_DEF
    else:
        for p in g2p(text):
            if p==' ':
                phonemes.append(SIL_DEF[0])
            p = re.sub(r'[0-9]', '', p)  # Remove stress
            if re.match(r'[A-Z]+', p):  # Only keep phonemes
                phonemes.append(p)
        #add one SIL symbol at the end so there's one at the end of each word
        phonemes.append('SIL')
    return phonemes

PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

PHONE_DEF_SIL = [
    'SIL', 'PAD','AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

CHANG_PHONE_DEF = [
    'AA', 'AE', 'AH', 'AW',
    'AY', 'B',  'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'P', 'R', 'S',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z'
]

CONSONANT_DEF = ['CH', 'SH', 'JH', 'R', 'B',
                 'M',  'W',  'V',  'F', 'P',
                 'D',  'N',  'L',  'S', 'T',
                 'Z',  'TH', 'G',  'Y', 'HH',
                 'K', 'NG', 'ZH', 'DH']

VOWEL_DEF = ['EY', 'AE', 'AY', 'EH', 'AA',
             'AW', 'IY', 'IH', 'OY', 'OW',
             'AO', 'UH', 'AH', 'UW', 'ER']

SIL_DEF = ['SIL']