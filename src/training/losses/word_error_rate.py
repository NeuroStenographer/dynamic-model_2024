from jiwer import wer
from torchaudio.pipelines import Tacotron2TTSBundle
from src.modeling.components.ctc_decoder import PhonemeCTCDecoder


def full_ctc_tts_pipeline(ctc_decoder, input_features, labels_map, reference_text, lexicon, tts_model=None):
    """
    Processes input features through a CTC decoder to text using a lexicon, optionally synthesizes the text to speech,
    and calculates the Word Error Rate (WER) against a reference text.

    Args:
        ctc_decoder (PhonemeCTCDecoder): The CTC decoder model.
        input_features (Tensor): The input features to the decoder.
        labels_map (dict): Mapping from model output indices to characters (including '<blank>').
        reference_text (str): The ground truth text for WER calculation.
        lexicon (dict): A dictionary mapping phoneme sequences to words.
        tts_model (Tacotron2TTSBundle, optional): The TTS model for speech synthesis. If None, TTS is skipped.

    Returns:
        str: The decoded text.
        Tensor, optional: The synthesized speech waveform if TTS model is provided. None otherwise.
        float: The Word Error Rate (WER).
    """
    # Decode phonemes to text using the lexicon
    decoded_text = decode_phonemes_to_text(ctc_decoder, input_features, labels_map, lexicon)

    # Optionally, use TTS to synthesize the decoded text into speech
    synthesized_speech = None
    if tts_model is not None:
        synthesized_speech = synthesize_text_to_speech(tts_model, decoded_text)

    # Calculate the Word Error Rate (WER)
    error_rate = wer(reference_text, decoded_text)

    return decoded_text, synthesized_speech, error_rate



def CTC_word_error_rate(logits, text):
    #TODO: uses ctc_decoder module to decode logits into text
    # and then calculate the word error rate
    pass


def word_error_rate(r, h):
    """
    Calculation of word_error_rate with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    r : list
    h : list
    Returns
    -------
    int
    Examples
    --------
    >>> word_error_rate("who is there".split(), "is there".split())
    1
    >>> word_error_rate("who is there".split(), "".split())
    3
    >>> word_error_rate("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def decode_phonemes_to_text(ctc_decoder, input_features, labels_map, lexicon):
    """
    Decodes input features (phonemes) into text using a CTC decoder and a lexicon.

    Args:
        ctc_decoder (nn.Module): The CTC decoder model.
        input_features (Tensor): The input features to the CTC decoder, representing phonemes or speech.
        labels_map (dict): A mapping from output indices of the CTC decoder to characters.
        lexicon (dict): A dictionary mapping phoneme sequences to words.

    Returns:
        str: The decoded text.
    """
    # Assuming the CTC decoder output is logits over character probabilities
    logits = ctc_decoder(input_features)
    decoded_indices = logits.argmax(dim=-1)  # Simplified greedy decoding
    phoneme_sequence = tuple(labels_map[idx] for idx in decoded_indices if idx != labels_map['<blank>'])

    # Convert phoneme sequence to words using the lexicon
    decoded_words = []
    current_phonemes = []
    for phoneme in phoneme_sequence:
        current_phonemes.append(phoneme)
        word = lexicon.get(tuple(current_phonemes))
        if word:
            decoded_words.append(word)
            current_phonemes = []  # Reset phoneme list after finding a word

    decoded_text = ' '.join(decoded_words)
    return decoded_text

def synthesize_text_to_speech(tts_model, decoded_text):
    """
    Converts decoded text into speech using a TTS model (e.g., Tacotron2).

    Args:
        tts_model (Tacotron2TTSBundle): The TTS model for speech synthesis.
        decoded_text (str): The text to synthesize.

    Returns:
        Tensor: The audio waveform of the synthesized speech.
    """
    waveform, _ = tts_model.tts(decoded_text)

    return waveform


def load_reference_text(file_path):
    """
    Load and return the content of a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        reference_text = file.read()
    return reference_text


def load_lexicon(filename):
    lexicon = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            phonemes = tuple(parts[1:])  # Create a tuple of the phoneme sequence
            lexicon[phonemes] = word
    return lexicon

# lexicon = load_lexicon('path_to_lexicon.txt')



def execute_ctc_tts_pipeline(ctc_decoder, input_features, labels_map, use_tts=False):
    lexicon_path = 'path_to_lexicon.txt'
    lexicon = load_lexicon(lexicon_path)

    reference_text_filepath = "path_to_reference_text.txt"
    reference_text = load_reference_text(reference_text_filepath)

    tts_model = Tacotron2TTSBundle() if use_tts else None

    decoded_text, synthesized_speech, error_rate = full_ctc_tts_pipeline(
        ctc_decoder=ctc_decoder, input_features=input_features,
        labels_map=labels_map, reference_text=reference_text, tts_model=tts_model, lexicon=lexicon)

    print(f"Decoded Text: {decoded_text}")
    print(f"Word Error Rate: {error_rate:.2%}")

    return decoded_text, synthesized_speech, error_rate


# Example usage
# input_features = ...  # Your input features here
# labels_map = ...  # Your mapping from indices to characters, including '<blank>'
# reference_text = "the expected transcription of the audio"
#
# # Execute the pipeline
# decoded_text, synthesized_speech, error_rate = execute_ctc_tts_pipeline(
#     ctc_decoder, input_features, labels_map, reference_text, use_tts=True)

# If TTS was used, synthesized_speech can now be saved or played
