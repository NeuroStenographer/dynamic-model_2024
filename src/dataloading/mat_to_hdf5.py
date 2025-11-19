import scipy
from pathlib import Path
import h5py
import numpy as np
import json
import traceback

from src.config import Config





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

def map_arrays(arrays, area, subarea):
    CH = len(arrays)
    T = arrays[0].shape[0]
    new_array = np.zeros((T, CH, 8, 8))
    for a, array in enumerate(arrays):
        channel_map = CHANNEL_MAPS[area][subarea]
        for i in range(8):
            for j in range(8):
                new_array[:,a,i,j] = array[:,channel_map[i][j]]
    return new_array

def mat_to_hdf5():
    # we will write a h5 file with the following structure:
    # partition / day / block / sentence / arraytype / area / time / (x, y)
    # partition / day / block / sentence / area / (time, arraytype, x, y) = (time, 5, 8, 8)
    #                                             ^^ like image / movie data
    cwd = Path.cwd()
    h5_path = cwd / 'data' / 'data.h5'
    h5_loc_strings = []
    sentence_start_timestep_idx = []
    sentence_end_timestep_idx = []
    block_start_timestep_idx = []
    day_start_timestep_idx = []
    try:
        with h5py.File(h5_path, 'w') as h5file:
            partitions = {'train': Path(Config.DIRS.MAT.TRAIN), 'test': Path(Config.DIRS.MAT.TEST), 'hold_out': Path(Config.DIRS.MAT.HOLD_OUT)}
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
        h5_path.unlink()
        raise e

if __name__ == "__main__":
    mat_to_hdf5()