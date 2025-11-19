import logging
from datetime import datetime
from src.runs import temporal_ica_run, simclr_run,  decoder_ctc_run, post_training_analysis
import psutil
import os
import torch
from src.modeling.dynamic_model import assemble_model
from pathlib import Path
import h5py
import scipy.io
import numpy as np
import json
from src.config import Config
import traceback
from src.dataloading.mat_to_hdf5 import map_arrays
from src.utils.phoneme_utils import text_to_phonemes, phoneme_to_id, clean_text, PHONE_DEF_SIL
from pathlib import Path
import scipy.io
import gc
import h5py
from src.dataloading.dataset import HDF5Dataset, MatDataset, SimpleTokenizer


def get_test_dataset(test_data_dir, batch_size, n_batches, get_phonemes=False):
    """

    Args:
        test_data_dir:

    Returns:test dataset

    """
    # if os.path.exists(test_data_dir):
    #     dataset = HDF5Dataset(test_data_dir, batch_size=batch_size, n_batches=n_batches, get_phonemes=get_phonemes)
    # else:
    # make h5 file using the MatDataset and then use DaskHDF5Dataset
    dataset = MatDataset(Config.DIRS.MAT.TEST)
    print("Created MatDataset for Further Use...")
    f = h5py.File(test_data_dir, 'w')
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
        # del dataset
        gc.collect()
        tokenizer = SimpleTokenizer()
        dataset = HDF5Dataset(test_data_dir, batch_size=batch_size,
                              tokenizer=tokenizer, n_batches=n_batches, get_phonemes=get_phonemes)
        for data in dataset:
            print(data['text'])
            print(data['token_ids'])
            print(data['token_lengths'])
    except Exception as e:
        f.close()
        # os.remove(test_data_dir)
        raise e
    return dataset


def mat_to_hdf5_test_data():
    # we will write a h5 file with the following structure:
    # partition / day / block / sentence / arraytype / area / time / (x, y)
    # partition / day / block / sentence / area / (time, arraytype, x, y) = (time, 5, 8, 8)
    #                                             ^^ like image / movie data
    base_dir = Path("G:\\competition_data\\competition\\test")
    h5_file_path = base_dir / "test_data.h5"
    h5_loc_strings = []
    sentence_start_timestep_idx = []
    sentence_end_timestep_idx = []
    block_start_timestep_idx = []
    day_start_timestep_idx = []
    try:
        with h5py.File(h5_file_path, 'w') as h5file:
            partitions = {'test': Path(Config.DIRS.MAT.TEST)}
            COUNT = 0 # total counter across partitions (train, test)
            for partition, partition_dir in partitions.items(): # train, test and hold_out
                # make a group for the partition
                h5file.create_group(partition)
                partition_paths = partition_dir.glob('*.mat')
                bad_days = []
                LEN = 0
                MAX_SENTENCE_LENGTH = 0
                print(f"Processing {partition} data")
                for p, path in enumerate(sorted(partition_paths)):
                    day_n = str(p)
                    session = path.stem
                    year, month, day = session.split('.')[1:]
                    date = f'{year}-{month}-{day}'
                    print(f"Processing {partition}: {date}")
                    try:
                        mat = scipy.io.loadmat(str(path))
                        # make a group for the date if successful load
                        h5file[partition].create_group(day_n)
                        # add metadata with the date
                        h5file[partition][day_n].attrs['date'] = date
                    except Exception as e:
                        bad_days.append(date)
                        print("=====================================")
                        print(f"Skipping {partition}: {date}")
                        print(f"Error loading {path}")
                        print(traceback.format_exc())
                        print("=====================================")
                        print()
                        continue
                    prev_block_num = -1
                    sentence_number = 0
                    day_start_timestep_idx.append(LEN)
                    for i, sentence in enumerate(mat['sentenceText']):
                        sentence = sentence.encode('utf-8')
                        block_number = mat['blockIdx'][i,0]
                        block_n = str(block_number)
                        # make a group for the block if it doesn't exist
                        if block_n not in h5file[partition][day_n]:
                            block_start_timestep_idx.append(LEN)
                            h5file[partition][day_n].create_group(block_n)
                            h5file[partition][day_n][block_n].attrs['block_number'] = block_number
                            h5file[partition][day_n][block_n].attrs['date'] = date
                        # now make a group for the sentence
                        if block_number != prev_block_num:
                            sentence_number = 0
                            prev_block_num = block_number
                        else:
                            sentence_number += 1
                        sentence_n = str(sentence_number)
                        # make a group for the sentence with block sample number
                        h5file[partition][day_n][block_n].create_group(sentence_n)
                        # add metadata with the sentence number
                        h5file[partition][day_n][block_n][sentence_n].attrs['sentence_number'] = sentence_number
                        h5file[partition][day_n][block_n][sentence_n].attrs['sentence_text'] = sentence
                        h5file[partition][day_n][block_n][sentence_n].attrs['block_number'] = block_number
                        h5file[partition][day_n][block_n][sentence_n].attrs['date'] = date
                        # get the arrays with normalized values
                        spike_pow = mat['spikePow'][0,i] # (time, channel)
                        tx1 = mat['tx1'][0,i] # (time, channel)
                        tx2 = mat['tx2'][0,i] # (time, channel)
                        tx3 = mat['tx3'][0,i] # (time, channel)
                        tx4 = mat['tx4'][0,i] # (time, channel)
                        arrays = [spike_pow, tx1, tx2, tx3, tx4]
                        session_sample_length = spike_pow.shape[0]
                        h5_loc_strings.append(f"{day_n}/{block_n}/{sentence_n}")
                        sentence_start_timestep_idx.append(LEN)
                        sentence_end_timestep_idx.append(LEN + session_sample_length)
                        # pin the sample group so we don't have to keep specifying it
                        sample_group = h5file[partition][day_n][block_n][sentence_n]
                        # UPDATE COUNT AND LEN FOR NEXT ITERATION
                        COUNT += 1
                        LEN += session_sample_length
                        MAX_SENTENCE_LENGTH = max(MAX_SENTENCE_LENGTH, session_sample_length)
                        for area in ['area_44', 'area_6v']:
                            # make a group for the area
                            sample_group.create_group(area)
                            # add metadata with the area
                            sample_group[area].attrs['area'] = area
                            # date, block, sentence, sentence_text
                            sample_group[area].attrs['date'] = date
                            sample_group[area].attrs['block_number'] = block_number
                            sample_group[area].attrs['sentence_number'] = sentence_number
                            sample_group[area].attrs['sentence_text'] = sentence
                            for subarea in ['superior', 'inferior']:
                                # make a group for the subarea
                                h5file[partition][day_n][block_n][sentence_n][area].create_group(subarea)
                                # add metadata with the subarea
                                sample_group[area][subarea].attrs['subarea'] = subarea
                                # date, block, sentence, sentence_text, area
                                sample_group[area][subarea].attrs['date'] = date
                                sample_group[area][subarea].attrs['block_number'] = block_number
                                sample_group[area][subarea].attrs['sentence_number'] = sentence_number
                                sample_group[area][subarea].attrs['sentence_text'] = sentence
                                sample_group[area][subarea].attrs['area'] = area
                                # map the arrays to the channel maps
                                array = map_arrays(arrays, area, subarea)
                                # make a dataset for the array
                                sample_group[area][subarea].create_dataset('data', data=array, dtype=np.float32)
                                print(f"Writing {partition} / {date} / {block_number} / {sentence_number} / {area} / {subarea}")
                                print(f"Shape: {sample_group[area][subarea]['data'].shape}")
                                # add metadata to the dataset at all relevant levels
                                for g in [sample_group, sample_group[area], sample_group[area][subarea]]:
                                    g.attrs['data_shape'] = array.shape
                                    sample_group.attrs['data_dims'] = ('time', 'arraytype', 'x', 'y')
                                    g.attrs['array_type_coords'] = ('spike_pow', 'tx1', 'tx2', 'tx3', 'tx4')
                                    g.attrs['x_coords'] = list(range(8))
                                    g.attrs['y_coords'] = list(range(8))
                                    # add sentence text to the metadata
                                    g.attrs['sentence_text'] = sentence
                partition = h5file[partition]
                # add the number of days to the metadata
                partition.attrs['n_days'] = len(partition.keys())
                # add the number of sentences to the metadata
                partition.attrs['n_sentences'] = COUNT
                # add the number of time steps to the metadata
                partition.attrs['n_time_steps'] = LEN
                # add the max sentence length to the metadata
                partition.attrs['max_sentence_length'] = MAX_SENTENCE_LENGTH
                # add the sentence start indexes to the metadata
                partition.attrs['sentence_start_timestep_idx'] = json.dumps(sentence_start_timestep_idx)
                # add the block start indexes to the metadata
                partition.attrs['block_start_timestep_idx'] = json.dumps(block_start_timestep_idx)
                # add the day start indexes to the metadata
                partition.attrs['day_start_timestep_idx'] = json.dumps(day_start_timestep_idx)
                # add the bad days to the metadata
                partition.attrs['bad_days'] = json.dumps(bad_days)
            # check that the metadata is of equal length
            # THESE ARE OF EQUAL LENGTH
            assert COUNT == len(h5_loc_strings) == len(sentence_start_timestep_idx) == len(sentence_end_timestep_idx), f"Error: Metadata not of equal length\n------\nCOUNT: {COUNT},\nlen(h5_loc_strings): {len(h5_loc_strings)},\nlen(sentence_start_timestep_idx): {len(sentence_start_timestep_idx)},\nlen(sentence_end_timestep_idx): {len(sentence_end_timestep_idx)}"

        h5file.close()

    except Exception as e:
        # delete the file if there is an error
        h5_file_path.unlink()
        raise e



def create_labels_map(alphabet):
    # Optionally, you might want to start with the blank symbol at index 0
    labels_map = {i: char for i, char in enumerate(alphabet)}
    labels_map[len(alphabet)] = '<blank>'  # Adjust according to your blank label handling
    return labels_map


def save_model(parts_from, model_dir):
    """

    Args:
        parts_from:
        model_dir:

    Returns:saved model

    """
    model_components = assemble_model(parts_from=parts_from)
    torch.save(model_components.state_dict(), model_dir)
    print("Model saved to {}".format(model_dir))
    model_components.eval()

    return model_components


def load_model(model_dir):
    """
    Load a trained PhonemeCTCDecoder model from a checkpoint.

    Args:
        model_path (str): Path to the model checkpoint file.

    Returns:
        model (torch.nn.Module): The loaded and ready-to-evaluate model.
    """
    entity = 'loudmouths' # your entity goes here
    project = 'decoder-ctc'
    parts_from = (
        ('frame_intake', entity, project, 'latest'),
        ('frame_encoder', entity, project, 'latest'),
        ('phoneme_decoder', entity, 'decoder-ctc', 'latest')
    )

    # Parameters used during the model's training
    model_components = assemble_model(parts_from=parts_from)

    # Load the model state from the checkpoint
    model_state = torch.load(model_dir, map_location=torch.device('cpu'))  # Adjust map_location as needed
    print("model state loaded")
    model_components.load_state_dict(model_state)
    model_components.eval()  # Set to evaluation mode

    return model_components


def setup_logging():
    dt = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        filename=f'logs/{dt}_training.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s')
    logger = logging.getLogger()
    # File handler - logs to a file
    file_handler = logging.FileHandler('logs/training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # Stream handler - logs to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

def initial_train_cycle():
    parts_from = ('frame_intake', 'frame_encoder', 'frame_projector2')
    entity, project, run_id = temporal_ica_run(
        n_epochs=1,
        n_batches=1,
        parts_from=parts_from,
        use_gpu=False)
    logging.info(temporal_ica_run.cache_info())
    temporal_ica_run.cache_clear()
    logging.info(temporal_ica_run.cache_info())
    parts_from = ('frame_intake', 'frame_encoder', 'frame_projector1')
    entity, project, run_id = simclr_run(
        n_epochs=1,
        n_batches=1,
        parts_from=parts_from,
        use_gpu=False)
    logging.info(simclr_run.cache_info())
    simclr_run.cache_clear()
    logging.info(simclr_run.cache_info())
    parts_from = ('frame_intake', 'frame_encoder', 'phoneme_decoder')
    entity, project, run_id = decoder_ctc_run(
        n_epochs=1,
        n_batches=2,
        batch_size=2,
        parts_from=parts_from,
        use_gpu=False
    )
    logging.info(simclr_run.cache_info())
    simclr_run.cache_clear()
    logging.info(simclr_run.cache_info())
    # close the main process
    print("Initial Training Cycle Complete.")

def full_training_cycle(prev_project, n_batches=1, n_ctc_passes=1, sim_func='cosine'):
    entity = 'loudmouths' # your entity goes here
    project = prev_project
    run_id = 'latest'
    # NOTE: using 'latest' for run_id will find the latest created folder. If you copy-paste a run into your project folder, it will be the latest run.
    # Parts from is a tuple of tuples. Each tuple is of the form:
    # (part_name, entity, project, run_id)
    parts_from = (
        ('frame_intake', entity, project, run_id),
        ('frame_encoder', entity, project, run_id),
        ('frame_projector2', entity, 'temporal-ica', 'latest')
    )
    entity, project, run_id = temporal_ica_run(
        n_epochs=1,
        n_batches=n_batches,
        sim_func=sim_func,
        parts_from=parts_from,
        use_gpu=False)
    logging.info(temporal_ica_run.cache_info())
    temporal_ica_run.cache_clear()
    logging.info(temporal_ica_run.cache_info())
    parts_from = (
        ('frame_intake', entity, project, run_id),
        ('frame_encoder', entity, project, run_id),
        ('frame_projector1', entity, 'simclr', 'latest')
    )
    entity, project, run_id = simclr_run(
        n_epochs=1,
        n_batches=n_batches,
        sim_func=sim_func,
        parts_from=parts_from,
        use_gpu=False)
    logging.info(simclr_run.cache_info())
    simclr_run.cache_clear()
    logging.info(simclr_run.cache_info())
    parts_from = (
        ('frame_intake', entity, project, 'latest'),
        ('frame_encoder', entity, project, 'latest'),
        ('phoneme_decoder', entity, 'decoder-ctc', 'latest')
    )
    entity, project, run_id = decoder_ctc_run(
        n_epochs=1,
        n_batches=n_batches,
        n_passes=n_ctc_passes,
        batch_size=2,
        parts_from=parts_from,
        use_gpu=False
    )

    return entity, project, run_id, parts_from



def generate_transcriptions(trained_model, eval_dataset):
    # Including uppercase, lowercase, digits, common punctuation, and a blank symbol
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:!? "
    labels_map = create_labels_map(alphabet)
    breakpoint()
    trancriptions = post_training_analysis(model=trained_model,
                                           dataset=eval_dataset,
                                           labels_map=labels_map, use_tts=True)

    return trancriptions




def main():
    p = psutil.Process(os.getpid())
    # Set the process to low priority
    p.nice(psutil.IDLE_PRIORITY_CLASS)
    #setup_logging()
    # initial_train_cycle()
    # entity, project, run_id, parts_from = full_training_cycle(prev_project='decoder-ctc', n_batches=20, n_ctc_passes=5, sim_func='cosine')
    # model_components = save_model(model_dir="G:/competition_data/competition_models/phoneme_recognition_20-05-2024_v1.pt", parts_from=parts_from)
    # print("Training Complete.")
    # logging.info(simclr_run.cache_info())
    # simclr_run.cache_clear()
    # logging.info(simclr_run.cache_info())
    #logging.shutdown()
    model = load_model(model_dir="D:/speechbci_data/competition_data/competition/competition_models/competition_models/phoneme_recognition_20-05-2024_v1.pt")
    print("Model loaded successfuly ")
    # mat_to_hdf5_test_data()
    test_dataset = get_test_dataset(test_data_dir="D:/speechbci_data/competition_data/test_h5/test_data.h5", n_batches=20,
                                    batch_size=1,
                                    get_phonemes=True)
    breakpoint()
    trancriptions = generate_transcriptions(trained_model=model,
                                            eval_dataset=test_dataset)
    print("Transcriptions Complete.")


    #return model_components, entity, project, run_id, trancriptions, model
    return trancriptions


if __name__ == "__main__":
    main()
    exit(code=1)
