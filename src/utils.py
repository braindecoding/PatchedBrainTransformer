import numpy as np
import random

import torch
from torch.utils.data import Dataset


from moabb.datasets import (
    AlexMI,
    BNCI2014001,
    BNCI2014004,
    BNCI2015001,
    BNCI2015004,
    Cho2017,
    Lee2019_MI,
    PhysionetMI,
)
from moabb.paradigms import MotorImagery


# ----------------------------------------------------------------------------------------------------------------------
# Load data from MOABB
def get_mindbigdata_eeg(
    file_path="datasets/EP1.01.txt", freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    Load MindBigData EEG dataset from local file - MNIST Visual Stimulus
    EP1.01.txt contains EEG data recorded while subjects viewed MNIST digits (0-9)
    """
    import pandas as pd
    import os

    if channels is None:
        # MindBigData EPOC uses 14 channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
        channels = [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
        ]

    if n_classes is None:
        n_classes = 10  # MNIST digits 0-9

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"MindBigData file not found: {file_path}")
            print("Falling back to MNIST data...")
            return None

        print(f"Loading MindBigData MNIST from: {file_path}")

        # Read the actual MindBigData file
        all_data_trials = []
        all_labels = []
        all_meta = []

        n_channels = len(channels)

        # Read file line by line
        with open(file_path, 'r') as f:
            lines = f.readlines()

        print(f"Found {len(lines)} lines in MindBigData file")

        # Parse MindBigData EPOC format
        # Format: [id] [event] [device] [channel] [code] [size] [data]
        # Fields separated by TAB, data separated by comma
        # Example: 67650	67636	EP	F7	7	260	4482.564102,4477.435897,4484.102564...
        trials_processed = 0
        max_trials = 5000  # Reduced from 10000 due to memory constraints

        # Group data by event_id to reconstruct multi-channel trials
        events_data = {}

        for line_idx, line in enumerate(lines):
            if trials_processed >= max_trials:
                break

            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue

            try:
                # Split the line by TAB (not comma!)
                parts = line.split('\t')
                if len(parts) < 6:  # Need at least id, event, device, channel, code, size
                    continue

                # Parse MindBigData EPOC format: [id] [event] [device] [channel] [code] [size] [data]
                record_id = parts[0].strip()
                event_id = parts[1].strip()
                device = parts[2].strip()
                channel = parts[3].strip()

                # Only process EPOC data
                if device != "EP":
                    continue

                # Extract digit label (code)
                try:
                    digit_label = int(float(parts[4]))
                    if not (0 <= digit_label <= 9):
                        continue  # Skip if not valid MNIST digit (skip -1 random signals)
                except:
                    continue  # Skip if can't parse digit

                # Extract size (actual number of samples in this signal)
                try:
                    signal_size = int(float(parts[5]))
                    if signal_size < 100:  # Skip if too few samples
                        continue
                except:
                    continue  # Skip if can't parse size

                # Extract EEG data (from column 6, comma-separated)
                if len(parts) >= 7:
                    data_str = parts[6].strip()
                    eeg_data_raw = []
                    for value_str in data_str.split(','):
                        try:
                            eeg_data_raw.append(float(value_str.strip()))
                        except:
                            continue
                else:
                    continue  # Skip if no data

                # Verify we have enough data points
                if len(eeg_data_raw) < 100:  # Need minimum samples
                    continue

                # Store data grouped by event
                if event_id not in events_data:
                    events_data[event_id] = {
                        'digit': digit_label,
                        'channels': {},
                        'signal_size': signal_size
                    }

                # Store channel data
                events_data[event_id]['channels'][channel] = np.array(eeg_data_raw)

            except Exception as e:
                continue  # Skip problematic lines

        # Convert events to trials
        print(f"Found {len(events_data)} events, processing into trials...")

        # Debug: Check signal sizes and channels per event
        signal_sizes = []
        channels_per_event = []
        for event_data in events_data.values():
            if 'signal_size' in event_data:
                signal_sizes.append(event_data['signal_size'])
            channels_per_event.append(len(event_data['channels']))

        if signal_sizes:
            print(f"Signal size stats: min={min(signal_sizes)}, max={max(signal_sizes)}, avg={sum(signal_sizes)/len(signal_sizes):.1f}")
        if channels_per_event:
            print(f"Channels per event: min={min(channels_per_event)}, max={max(channels_per_event)}, avg={sum(channels_per_event)/len(channels_per_event):.1f}")
            print(f"Events with 14 channels: {sum(1 for x in channels_per_event if x == 14)}/{len(channels_per_event)}")

        for event_id, event_data in events_data.items():
            if trials_processed >= max_trials:
                break

            # Check if we have enough channels for this event
            available_channels = list(event_data['channels'].keys())
            if len(available_channels) < 10:  # Need at least 10 channels for good quality
                continue

            # Map available channels to our expected EPOC layout
            trial_data_list = []
            used_channels = []

            for expected_ch in channels:
                if expected_ch in available_channels:
                    trial_data_list.append(event_data['channels'][expected_ch])
                    used_channels.append(expected_ch)
                elif len(available_channels) > 0:
                    # Use any remaining channel as fallback
                    fallback_ch = available_channels[0]
                    trial_data_list.append(event_data['channels'][fallback_ch])
                    used_channels.append(fallback_ch)
                    available_channels.remove(fallback_ch)
                else:
                    # Not enough channels, skip this event
                    break

            if len(trial_data_list) < n_channels:
                continue  # Skip if not enough channels

            # Ensure all channels have same length
            channel_lengths = [len(data) for data in trial_data_list]
            min_length = min(channel_lengths)
            max_length = max(channel_lengths)

            if min_length < 100:  # Need minimum samples
                continue

            # Use consistent length for all trials
            # EPOC captures ~256 samples for 2 seconds (128Hz actual rate)
            # But actual samples can vary, so we standardize to 256
            target_samples = 256  # Standard EPOC length
            n_samples_to_use = min(min_length, target_samples)

            # Create trial data matrix with fixed size
            trial_data = np.zeros((n_channels, target_samples))

            for ch_idx in range(n_channels):
                if ch_idx < len(trial_data_list):
                    channel_data = trial_data_list[ch_idx][:n_samples_to_use]

                    # Always resample to target length for consistency
                    if len(channel_data) != target_samples:
                        trial_data[ch_idx] = np.interp(
                            np.linspace(0, 1, target_samples),
                            np.linspace(0, 1, len(channel_data)),
                            channel_data
                        )
                    else:
                        trial_data[ch_idx] = channel_data

            # EPOC data is already in microvolts, no need to scale
            # Just normalize to reasonable range
            trial_data = trial_data / 1000.0  # Convert to millivolts for better numerical stability

            all_data_trials.append(trial_data)
            all_labels.append(event_data['digit'])

            # Create meta info
            meta_info = {
                'subject': 1,
                'session': 'session_1',
                'run': 1,
                'event_id': event_id,
                'digit': event_data['digit'],
                'channels_used': used_channels[:n_channels]
            }
            all_meta.append(meta_info)

            trials_processed += 1

            if trials_processed % 100 == 0:
                print(f"Processed {trials_processed} trials...")

        print(f"Successfully processed {trials_processed} complete trials from {len(events_data)} events")

        if len(all_data_trials) == 0:
            print("No valid trials found in MindBigData file, using synthetic data...")
            return None

        # Convert to numpy arrays
        data_array = np.array(all_data_trials)
        all_labels = np.array(all_labels, dtype=int)
        meta_df = pd.DataFrame(all_meta)

        print(f"Successfully loaded {len(all_data_trials)} MNIST trials with shape {data_array.shape}")
        print(f"MNIST digit distribution: {np.bincount(all_labels)}")

        return data_array, all_labels, meta_df, channels

    except Exception as e:
        print(f"Error loading MindBigData file: {e}")
        print("Falling back to synthetic MNIST data...")
        return None




def get_AlexMI(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    PhD-Theses (french): https://theses.hal.science/tel-01196752
    data: https://zenodo.org/records/806023

    Electrode montage: corresponding to the international 10-20 system
    """

    if channels is None:
        channels = [
            "Fpz",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
        ]

    # Labels: right_hand, feet, rest
    if n_classes is None:
        n_classes = 3

    if subject is None:
        subject = list(range(1, 9))

    dataset = AlexMI()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels[np.where(labels == "rest")] = 4
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_BNCI2014001(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://lampx.tugraz.at/~bci/database/001-2014/description.pdf
    This four class motor imagery data set was originally released as data set 2a of the BCI Competition IV


    Electrode montage: corresponding to the international 10-20 system
    """

    if channels is None:
        channels = [
            "Fz",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "P1",
            "Pz",
            "P2",
            "POz",
        ]

    # Labels: left_hand, right_hand, feet, tongue
    if n_classes is None:
        n_classes = 4

    if subject is None:
        subject = list(range(1, 10))

    dataset = BNCI2014001()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels[np.where(labels == "tongue")] = 3
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_BNCI2014004(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://lampx.tugraz.at/~bci/database/004-2014/description.pdf
    https://ieeexplore.ieee.org/document/4359220

    Electrode montage: 3 bipolar channels (C3, Cz, C4) placed according to the extended 10-20 system
    """

    if channels is None:
        channels = ["C3", "Cz", "C4"]

    # Labels: left_hand, right_hand
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 10))

    dataset = BNCI2014004()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_BNCI2015001(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://lampx.tugraz.at/~bci/database/001-2015/description.pdf

    Electrode montage: 13 channels placed according to the 10-20 system
    """

    if channels is None:
        channels = [
            "FC3",
            "FCz",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CPz",
            "CP4",
        ]

    # Labels: right_hand, feet
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 13))

    dataset = BNCI2015001()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_BNCI2015004(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://lampx.tugraz.at/~bci/database/004-2015/description.pdf

    Electrode montage: 13 channels placed according to the 10-20 system
    """

    if channels is None:
        channels = [
            "AFz",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "FC3",
            "FCz",
            "FC4",
            "T3",
            "C3",
            "Cz",
            "C4",
            "T4",
            "CP3",
            "CPz",
            "CP4",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO3",
            "PO4",
            "O1",
            "O2",
        ]

    # Labels: right_hand, feet
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 10))

    dataset = BNCI2015004()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    # drop trials with label 'word_ass' 'subtraction', 'navigation'
    idx = np.concatenate(
        (np.where(labels == "feet")[0], np.where(labels == "right_hand")[0])
    )
    data = data[idx]
    labels = labels[idx]
    meta = meta.iloc[idx]

    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_Cho2017(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://academic.oup.com/gigascience/article/6/7/gix034/3796323

    Electrode montage: 64 channels placed according to the 10-10 system
    """

    if channels is None:
        channels = [
            "Fp1",
            "AF7",
            "AF3",
            "F1",
            "F3",
            "F5",
            "F7",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "C1",
            "C3",
            "C5",
            "T7",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "P1",
            "P3",
            "P5",
            "P7",
            "P9",
            "PO7",
            "PO3",
            "O1",
            "Iz",
            "Oz",
            "POz",
            "Pz",
            "CPz",
            "Fpz",
            "Fp2",
            "AF8",
            "AF4",
            "AFz",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT8",
            "FC6",
            "FC4",
            "FC2",
            "FCz",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP8",
            "CP6",
            "CP4",
            "CP2",
            "P2",
            "P4",
            "P6",
            "P8",
            "P10",
            "PO8",
            "PO4",
            "O2",
        ]

    # Labels: left_hand, right_hand
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 53))
        # ValueError: Invalid subject 32, 46, 49
        subject.remove(32)
        subject.remove(46)
        subject.remove(49)

    dataset = Cho2017()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_Lee2019_MI(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://academic.oup.com/gigascience/article/6/7/gix034/3796323

    64 channels placed according to the 10-10 system
    """

    if channels is None:
        channels = [
            "AF3",
            "AF4",
            "AF7",
            "AF8",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "CP1",
            "CP2",
            "CP3",
            "CP4",
            "CP5",
            "CP6",
            "CPz",
            "Cz",
            "F10",
            "F3",
            "F4",
            "F7",
            "F8",
            "F9",
            "FC1",
            "FC2",
            "FC3",
            "FC4",
            "FC5",
            "FC6",
            "FT10",
            "FT9",
            "Fp1",
            "Fp2",
            "Fz",
            "O1",
            "O2",
            "Oz",
            "P1",
            "P2",
            "P3",
            "P4",
            "P7",
            "P8",
            "PO10",
            "PO3",
            "PO4",
            "PO9",
            "POz",
            "Pz",
            "T7",
            "T8",
            "TP10",
            "TP7",
            "TP9",
        ]

    # Labels: left_hand, right_hand
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 55))

    dataset = Lee2019_MI()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_PhysionetMI(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://academic.oup.com/gigascience/article/6/7/gix034/3796323

    Electrode montage: 64 electrodes as per the international 10-10 system
    (excluding electrodes Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, and P10)
    """

    if channels is None:
        channels = [
            "Fp1",
            "Fpz",
            "Fp2",
            "AF7",
            "AF3",
            "AFz",
            "AF4",
            "AF8",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "FT8",
            "T9",
            "T7",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "T10",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "O1",
            "Oz",
            "O2",
            "Iz",
        ]

    # Labels: left_hand, right_hand
    if n_classes is None:
        n_classes = 4

    if subject is None:
        subject = list(range(1, 110))

    dataset = PhysionetMI()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels[np.where(labels == "rest")] = 4
    # booth hands -> soft labels? [.5, .5, 0, 0, 0]
    labels[np.where(labels == "hands")] = 5

    labels = labels.astype(int)
    # data[:, np.array([19, 21, 23, 28, 29, 30, 31, 32, 33, 34, 39, 41, 43])]
    return data, labels, meta, channels


# ----------------------------------------------------------------------------------------------------------------------
# Data loader


class SeqDataset(Dataset):
    def __init__(
        self,
        dim_token,
        num_tokens_per_channel,
        reduce_num_chs_to=False,
        augmentation=[],
    ):

        self.num_tokens_per_channel = num_tokens_per_channel
        self.dim_token = dim_token

        self.list_data_sets = []
        self.list_channel_names = []
        self.list_labels = []

        # list of tuples with (trial_data, trial_label, index_data_set)
        self.list_trials = []

        self.int_pos_channels_per_data_set = []
        self.dict_channels = {}

        # if cls-token should be learnable or not zero it can be overwritten by the model
        self.cls = torch.zeros(1, dim_token)

        # drop random input tokens
        self.reduce_num_chs_to = reduce_num_chs_to

        if (
            len(
                set(augmentation)
                - {"time_shifts", "DC_shifts", "amplitude_scaling", "noise"}
            )
            != 0
        ):
            no_aug = str(
                set(augmentation)
                - {"time_shifts", "DC_shifts", "amplitude_scaling", "noise"}
            )
            raise ValueError(no_aug + " is not supported as data augmentation")
        self.augmentation = augmentation

    def append_data_set(self, data_set, channel_names, label):
        """
        Note: All data is loaded into RAM, which can be a problem with large amounts of data.
              If it fits, it's faster.

        data_set: np.array of size Trials x Channels x Time
        channel_names: list
        label: np.array of size Trials
        """

        if data_set.shape[0] == label.shape[0] and data_set.shape[1] == len(
            channel_names
        ):
            self.list_data_sets += [data_set]
            self.list_channel_names += [channel_names]
            self.list_labels += [label]
        else:
            raise ValueError("Append data set is not possible, size does not match!")

    def prepare_data_set(self, set_pos_channels=None):
        """
        set_pos_channels (dictionary int_pos_channels_per_data_set): to copy int. channel position from existing
        Dataset (e.g. to ensure train and test datasets return the same position)

        list_trial = list of tuples with (trial_data, trial_label, index_data_set), all as tensors
        """

        self.list_trials = [
            (
                torch.from_numpy(data[idx]).float(),
                torch.LongTensor([label[idx]]),
                torch.LongTensor([idx_ds]),
            )
            for idx_ds, (data, label) in enumerate(
                zip(self.list_data_sets, self.list_labels)
            )
            for idx in range(data.shape[0])
        ]

        unique_channel_names = list(np.unique(sum(self.list_channel_names, [])))

        if set_pos_channels is not None:
            # check if there are new channels:
            new_channels = list(
                set(unique_channel_names) - set(set_pos_channels.keys())
            )
            if len(new_channels) == 0:
                self.dict_channels = set_pos_channels
            else:
                print("Following new channels are added: " + str(new_channels))
                raise ValueError("There are some new Channels")

        else:
            # CLS token has always position 0 -> pos channel start at 1
            self.dict_channels = {
                key: torch.IntTensor(
                    [
                        *range(
                            i * self.num_tokens_per_channel + 1,
                            (i + 1) * self.num_tokens_per_channel + 1,
                        )
                    ]
                )
                for i, key in enumerate(unique_channel_names)
            }

        self.int_pos_channels_per_data_set = [
            torch.cat(
                ([self.dict_channels[key].unsqueeze(dim=0) for key in channel_names]),
                dim=0,
            )
            for channel_names in self.list_channel_names
        ]

        labels = np.array([int(trial[1]) for trial in self.list_trials])
        num_labels = [(i, np.where(labels == i)[0].shape) for i in set(labels)]
        print(num_labels)

        # free some memory
        # self.list_data_sets, self.list_channel_names, self.list_labels = None, None, None

    def __len__(self):
        return len(self.list_trials)

    def __getitem__(self, idx):
        """
        dim_time size: #token x dim batch
        label size: dim batch
        int_pos size: dim batch
        """

        dim_time = self.num_tokens_per_channel * self.dim_token

        if "time_shifts" in self.augmentation:
            data = torch.cat(
                (
                    self.cls,
                    self.list_trials[idx][0][
                        :,
                        (
                            st := random.randint(
                                0, self.list_trials[idx][0].shape[1] - dim_time - 1
                            )
                        ) : st
                        + dim_time,
                    ].reshape(-1, self.dim_token),
                ),
                dim=0,
            )
        else:
            st = (self.list_trials[idx][0].shape[1] - dim_time - 1) // 2
            data = torch.cat(
                (
                    self.cls,
                    self.list_trials[idx][0][:, st : st + dim_time].reshape(
                        -1, self.dim_token
                    ),
                ),
                dim=0,
            )

        if "DC_shifts" in self.augmentation:
            data += torch.rand(1) * 0.2 - 0.1

        if "amplitude_scaling" in self.augmentation:
            data *= torch.rand(1) * 0.2 + 0.9

        if "noise" in self.augmentation:
            data += torch.normal(mean=0, std=0.1, size=data.size())

        label = self.list_trials[idx][1]

        # cls-token has pos. 0
        int_pos = torch.cat(
            (
                torch.IntTensor([0]),
                self.int_pos_channels_per_data_set[self.list_trials[idx][2]][
                    :, : self.num_tokens_per_channel
                ].flatten(),
            ),
            dim=0,
        )

        if self.reduce_num_chs_to and data.size(0) > self.reduce_num_chs_to:
            idx_channels = np.arange(1, data.size(0) // self.num_tokens_per_channel)
            np.random.shuffle(idx_channels)
            idx_channels = idx_channels[: self.reduce_num_chs_to]
            idx = np.array(
                sum(
                    [
                        list(
                            range(
                                i * self.num_tokens_per_channel,
                                i * self.num_tokens_per_channel
                                + self.num_tokens_per_channel,
                            )
                        )
                        for i in idx_channels
                    ],
                    [],
                )
            )
            idx = np.concatenate((np.array([0]), idx))

            return data[idx], label, int_pos[idx]
        else:
            #  data: tensor, label: tensor, int_pos: tensor
            return data, label, int_pos

    @staticmethod
    def my_collate(batch):
        """
        Converts the output of the generator into the appropriate form
        https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
        """

        num_token_per_trial = [item[0].size(0) for item in batch]
        unique_num_token_within_batch = sorted(set(num_token_per_trial))
        data = [
            torch.empty((0, num_tok, batch[0][0].size(1)))
            for num_tok in unique_num_token_within_batch
        ]
        label = [torch.empty(0) for num_tok in unique_num_token_within_batch]
        int_pos = [
            torch.empty((0, num_tok)) for num_tok in unique_num_token_within_batch
        ]
        unique_num_token_within_batch = np.array(list(unique_num_token_within_batch))
        mini_batch_idx = [
            np.where(unique_num_token_within_batch == num_tok)[0][0]
            for num_tok in num_token_per_trial
        ]

        for i, item in enumerate(batch):
            data[mini_batch_idx[i]] = torch.cat(
                (data[mini_batch_idx[i]], item[0].unsqueeze(0)), dim=0
            )
            label[mini_batch_idx[i]] = torch.cat(
                (label[mini_batch_idx[i]], item[1]), dim=0
            )
            int_pos[mini_batch_idx[i]] = torch.cat(
                (int_pos[mini_batch_idx[i]], item[2].unsqueeze(0)), dim=0
            )

        return {"patched_eeg_token": data, "labels": label, "pos_as_int": int_pos}


# ----------------------------------------------------------------------------------------------------------------------
# scaling methods
def scale(mne_epochs):
    return (mne_epochs - np.mean(mne_epochs, axis=2, keepdims=True)) / (
        np.max(mne_epochs, axis=2, keepdims=True)
        - np.min(mne_epochs, axis=2, keepdims=True)
    )


def zero_mean_unit_var(mne_epochs, meta_data):
    for sub in list(set(meta_data["subject"])):
        for session in list(set(meta_data["session"])):
            data = mne_epochs[
                np.where(
                    (meta_data["subject"] == sub) & (meta_data["session"] == session)
                )
            ]
            mne_std = (
                data.transpose(1, 0, 2)
                .reshape(data.shape[1], data.shape[0] * data.shape[2])
                .std(axis=1)
            )
            mne_mean = (
                data.transpose(1, 0, 2)
                .reshape(data.shape[1], data.shape[0] * data.shape[2])
                .mean(axis=1)
            )
            mne_std = np.expand_dims(mne_std, axis=1)
            mne_mean = np.expand_dims(mne_mean, axis=1)
            data = (data - mne_mean) / mne_std
            mne_epochs[
                np.where(
                    (meta_data["subject"] == sub) & (meta_data["session"] == session)
                )
            ] = data

    return mne_epochs


# ----------------------------------------------------------------------------------------------------------------------
# train test split
def train_test_split(data, labels, meta, test_size=0.05):
    """
    Stratified train-test split untuk memastikan distribusi label seimbang
    """
    from sklearn.model_selection import train_test_split as sklearn_split

    # Gunakan stratified split untuk mempertahankan distribusi label
    train_idx, test_idx = sklearn_split(
        np.arange(data.shape[0]),
        test_size=test_size,
        stratify=labels,
        random_state=42  # Untuk reproduksibilitas
    )

    # Validasi tidak ada overlap antara train dan test indices
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Data leakage: overlapping indices in train/test split!"
    assert len(train_idx) + len(test_idx) == data.shape[0], "Sample count mismatch after split!"

    train_data = data[train_idx]
    train_labels = labels[train_idx]
    train_meta = meta.iloc[train_idx].reset_index(drop=True)
    # Simpan original indices untuk validasi
    train_meta['original_idx'] = train_idx

    test_data = data[test_idx]
    test_labels = labels[test_idx]
    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    # Simpan original indices untuk validasi
    test_meta['original_idx'] = test_idx

    # Print distribusi untuk verifikasi
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")
    print(f"Split validation: {len(train_idx)} train + {len(test_idx)} test = {len(train_idx) + len(test_idx)} total samples")

    return train_data, train_labels, train_meta, test_data, test_labels, test_meta
