import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import scipy
from torch.utils.data import DataLoader


AUDIO_DIR = "birdclef-2025/train_audio"
ANNOTATIONS_FILE = "birdclef-2025/train.csv"
NUM_SAMPLES = 22050
TARGET_SAMPLE_RATE = 22050
BATCH_SIZE = 16


class BirdClefDataset(Dataset):

    def __init__(self, audio_dir, annotation_dir, device, num_samples, target_sample_rate, transformation=None):
        super().__init__()
        self.device = device
        self.audio_dir = audio_dir
        self.annotation_file = self._get_annotation_file(annotation_dir)
        if transformation is not None:
            self.transformation = transformation.to(self.device)
        self.num_samples = num_samples
        self.target_sample_rate = target_sample_rate

        self.label_encoder = LabelEncoder()
        self.labels = self.annotation_file.iloc[:, 0]
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._bandpass_filtering(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._padding_if_necessary(signal)
        if self.transformation is not None:
            signal = self.transformation(signal)

        return signal, label
    
    def _padding_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dimemsion_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dimemsion_padding)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _bandpass_filtering(self, signal, sr, freq_range=(300, 8000), order=5):
        signal_np = signal.cpu().numpy()
        sos = scipy.signal.butter(order, freq_range, "bandpass", fs=sr, output="sos")
        filtered_signal_np = scipy.signal.sosfiltfilt(sos, signal_np)
        filtered_signal = torch.tensor(filtered_signal_np, dtype=torch.float32).to(self.device)
        return filtered_signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _get_annotation_file(self, annotation_dir):
        annotation_file = pd.read_csv(annotation_dir)
        return annotation_file

    def _get_audio_sample_label(self, index):
        encoded_label = self.encoded_labels[index]
        one_hot_encoded_label = F.one_hot(torch.tensor(encoded_label), num_classes = self.num_classes).float().to(self.device)
        return one_hot_encoded_label

    def _get_audio_sample_path(self, index):
        path = os.path.join(AUDIO_DIR, self.labels[index])
        return path



if __name__ == "__main__":

    
    if torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device = {device}")

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=NUM_SAMPLES,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )


    birdDataset = BirdClefDataset(AUDIO_DIR, ANNOTATIONS_FILE, device, NUM_SAMPLES, TARGET_SAMPLE_RATE, mel_spectogram)

    bird_dataset = DataLoader(birdDataset, batch_size=BATCH_SIZE)
    print(len(bird_dataset))
    