"""
DataLoader for SEGLVIK Fin-whale Dataset
Designed for CPC unsupervised pre-training

Key features:
- Simple sliding window sampling
- Only returns raw audio (no spectrograms)
- Supports train/val/test splits
- Memory efficient (loads audio on-the-fly)
"""
import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm


class SEGLVIKDataset(Dataset):
    """
    Simple SEGLVIK dataset for CPC training
    
    Args:
        data_folder: Path to SEGLVIK folder
        split: 'train', 'val', or 'test'
        window_duration_sec: Window size in seconds (e.g., 8.0)
        hop_duration_sec: Hop size in seconds (e.g., 0.5)
        sample_rate: Target sample rate (e.g., 16000)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        data_folder,
        split="train",
        window_duration_sec=8.0,
        hop_duration_sec=0.5,
        sample_rate=16000,
        seed=42
    ):
        super().__init__()
        
        # Set random seed
        torch.manual_seed(seed)
        
        # Store parameters
        self.data_folder = data_folder
        self.split = split
        self.window_duration_sec = window_duration_sec
        self.hop_duration_sec = hop_duration_sec
        self.target_sample_rate = sample_rate
        self.seed = seed
        
        # Determine split folder
        split_folders = {
            'train': 'trainset',
            'val': 'validset',
            'test': 'testset'
        }
        
        if split not in split_folders:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")
        
        split_folder = split_folders[split]
        self.audio_dir = os.path.join(data_folder, split_folder, 'flacs')
        
        # Check if folder exists
        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError(f"Audio folder not found: {self.audio_dir}")
        
        # Load audio files
        self.audio_files = sorted(glob(os.path.join(self.audio_dir, "*.flac")))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No .flac files found in {self.audio_dir}")
        
        print(f"\n{'='*60}")
        print(f"SEGLVIK Dataset - {split.upper()}")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Audio folder: {self.audio_dir}")
        print(f"Audio files: {len(self.audio_files)}")
        print(f"Window duration: {window_duration_sec} s")
        print(f"Hop duration: {hop_duration_sec} s")
        print(f"Target sample rate: {sample_rate} Hz")
        
        # Get sample rate from first file
        first_file = self.audio_files[0]
        self.original_sample_rate = sf.info(first_file).samplerate
        print(f"Original sample rate: {self.original_sample_rate} Hz")
        
        # Calculate window and hop sizes in samples
        self.window_size = int(window_duration_sec * self.original_sample_rate)
        self.hop_size = int(hop_duration_sec * self.original_sample_rate)
        
        # Prepare samples (file, start_sample pairs)
        print(f"\nPreparing samples...")
        self._prepare_samples()
        
        print(f"Total samples: {len(self.samples)}")
        print(f"{'='*60}\n")
    
    def _prepare_samples(self):
        """
        Create a list of (file_path, start_sample, file_idx) tuples
        using sliding window approach
        """
        self.samples = []
        
        for file_idx, audio_file in enumerate(tqdm(self.audio_files, desc="Indexing audio")):
            # Get audio length without loading the full file
            info = sf.info(audio_file)
            num_samples = info.frames
            
            # Create windows with sliding hop
            for start in range(0, num_samples - self.window_size + 1, self.hop_size):
                self.samples.append((audio_file, start, file_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and return a single audio window
        
        Returns:
            dict with keys:
                - 'raw_audio': torch.Tensor of shape [T] (raw waveform)
                - 'file_name': str (filename)
                - 'start_time': float (start time in seconds)
                - 'file_idx': int (file index)
        """
        audio_file, start_sample, file_idx = self.samples[idx]
        
        # Load audio segment using soundfile (more reliable)
        audio_np, sr = sf.read(
            audio_file,
            start=start_sample,
            frames=self.window_size,
            dtype='float32'
        )
        
        # Convert to torch tensor
        audio = torch.from_numpy(audio_np)
        
        # Convert to mono if stereo
        if audio.dim() > 1 and audio.shape[-1] > 1:
            audio = audio.mean(dim=-1)
        
        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        
        # Create sample dictionary
        sample = {
            'raw_audio': audio,
            'file_name': os.path.basename(audio_file),
            'start_time': start_sample / sr,
            'file_idx': file_idx,
        }
        
        return sample


def create_dataloaders(
    data_folder,
    split="train",
    window_duration_sec=8.0,
    hop_duration_sec=0.5,
    sample_rate=16000,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    seed=42,
):
    """
    Create DataLoader for SEGLVIK dataset
    
    Args:
        data_folder: Path to SEGLVIK root folder
        split: 'train', 'val', or 'test'
        window_duration_sec: Window size in seconds
        hop_duration_sec: Hop size in seconds
        sample_rate: Target sample rate
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        pin_memory: Whether to pin memory (faster GPU transfer)
        seed: Random seed
    
    Returns:
        DataLoader instance
    
    Example:
        >>> train_loader = create_dataloaders(
        ...     data_folder="/path/to/SEGLVIK",
        ...     split="train",
        ...     window_duration_sec=8.0,
        ...     batch_size=32,
        ... )
        >>> for batch in train_loader:
        ...     raw_audio = batch['raw_audio']  # [B, T]
        ...     # Train your model
    """
    
    # Create dataset
    dataset = SEGLVIKDataset(
        data_folder=data_folder,
        split=split,
        window_duration_sec=window_duration_sec,
        hop_duration_sec=hop_duration_sec,
        sample_rate=sample_rate,
        seed=seed
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )
    
    return dataloader


# ============= Supervised DataLoader (for validation with labels) =============

def get_random_negative_part(signal_length, detections, max_size_selection, max_try=10):
    """
    Find a random audio segment that doesn't contain any whale pulses
    
    Args:
        signal_length: Total number of samples in audio
        detections: List of (start, end) tuples for pulse locations (in samples)
        max_size_selection: Window size in samples
        max_try: Maximum attempts to find a negative segment
    
    Returns:
        (start_idx, end_idx) or (None, None) if not found
    """
    def is_partially_contained(interval_a, interval_b):
        """Check if two intervals overlap"""
        a1, a2 = interval_a
        b1, b2 = interval_b
        return (b1 <= a1 <= b2) or (b1 <= a2 <= b2) or (a1 <= b1 <= a2) or (a1 <= b2 <= a2)
    
    current_try = 0
    random_index_start = torch.randint(
        max_size_selection // 2, 
        signal_length - max_size_selection,
        (1,)
    ).item()
    random_index_end = random_index_start + max_size_selection
    
    while current_try < max_try:
        # Check if this window overlaps with any detection
        overlaps = any(
            is_partially_contained(detection_time, [random_index_start, random_index_end])
            for detection_time in detections
        )
        
        if not overlaps:
            return random_index_start, random_index_end
        
        # Try again
        random_index_start = torch.randint(
            0,
            signal_length - max_size_selection,
            (1,)
        ).item()
        random_index_end = random_index_start + max_size_selection
        current_try += 1
    
    return None, None


class SEGLVIKSupervisedDataset(Dataset):
    """
    SEGLVIK dataset with labels for supervised evaluation
    
    Generates balanced positive/negative samples:
    - Positive samples: Windows containing whale pulses (label=1)
    - Negative samples: Windows without whale pulses (label=0)
    
    Args:
        data_folder: Path to SEGLVIK folder
        split: 'val' or 'test'
        window_duration_sec: Window size in seconds
        sample_rate: Target sample rate
        tfr_by_pulse: Number of windows per pulse (for positives)
        confidence_threshold: Minimum confidence score (optional)
        snr_threshold: Minimum SNR (optional)
        seed: Random seed
    """
    
    def __init__(
        self,
        data_folder,
        split="val",
        window_duration_sec=8.0,
        sample_rate=16000,
        tfr_by_pulse=5,
        confidence_threshold=None,
        snr_threshold=None,
        seed=42
    ):
        super().__init__()
        
        torch.manual_seed(seed)
        
        # Store parameters
        self.data_folder = data_folder
        self.split = split
        self.window_duration_sec = window_duration_sec
        self.target_sample_rate = sample_rate
        self.tfr_by_pulse = tfr_by_pulse
        self.seed = seed
        
        split_folders = {
            'val': 'validset',
            'test': 'testset'
        }
        
        if split not in split_folders:
            raise ValueError(f"Supervised split must be 'val' or 'test'")
        
        split_folder = split_folders[split]
        split_path = os.path.join(data_folder, split_folder)
        self.audio_dir = os.path.join(split_path, 'flacs')
        csv_path = os.path.join(split_path, 'detections.csv')
        
        print(f"\n{'='*60}")
        print(f"SEGLVIK Supervised Dataset - {split.upper()}")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Audio folder: {self.audio_dir}")
        print(f"CSV: {csv_path}")
        
        # Load annotations
        import pandas as pd
        self.annotations = pd.read_csv(csv_path)
        print(f"Annotations loaded: {len(self.annotations)} detections")
        
        # Load audio files
        audio_files = glob(os.path.join(self.audio_dir, "*.flac"))
        self.audio_file_dict = {os.path.basename(f): f for f in audio_files}
        print(f"Audio files: {len(self.audio_file_dict)}")
        
        # Get sample rate
        first_file = list(self.audio_file_dict.values())[0]
        self.original_sample_rate = sf.info(first_file).samplerate
        print(f"Sample rate: {self.original_sample_rate} Hz")
        print(f"Window duration: {window_duration_sec} s")
        print(f"TFR by pulse: {tfr_by_pulse}")
        
        # Generate positive and negative samples
        pos_samples, num_pos = self._generate_pos_samples(
            confidence_threshold=confidence_threshold,
            snr_threshold=snr_threshold
        )
        
        neg_samples, num_neg = self._generate_neg_samples(
            num_neg_samples=num_pos
        )
        
        # Combine samples
        self.samples = []
        for i in range(min(len(pos_samples), len(neg_samples))):
            self.samples.append((pos_samples[i], 1.0))  # Positive
            self.samples.append((neg_samples[i], 0.0))   # Negative
        
        print(f"Total samples: {len(self.samples)} ({num_pos} pos + {num_neg} neg)")
        print(f"{'='*60}\n")
    
    def _generate_pos_samples(self, confidence_threshold=None, snr_threshold=None):
        """Generate positive samples (with whale pulses)"""
        print(f"\nGenerating positive samples...")
        
        annots = self.annotations.copy()
        
        # Apply filters
        if confidence_threshold is not None:
            annots = annots[annots["conf"] >= confidence_threshold]
            print(f"  After confidence filter: {len(annots)}")
        
        if snr_threshold is not None:
            if "snr" in annots.columns:
                annots = annots[annots["snr"] >= snr_threshold]
                print(f"  After SNR filter: {len(annots)}")
        
        pos_samples = []
        annots_grouped = annots.groupby("filename")
        
        for filename, group in tqdm(annots_grouped, desc="  Processing pulses"):
            if filename not in self.audio_file_dict:
                continue
            
            file_path = self.audio_file_dict[filename]
            audio_info = sf.info(file_path)
            audio_length = audio_info.frames
            
            for _, annot in group.iterrows():
                start_pulse = int(annot["Begin Time (s)"] * self.original_sample_rate)
                end_pulse = int(annot["End Time (s)"] * self.original_sample_rate)
                
                # Generate multiple windows per pulse
                shifts = self._generate_shifts(
                    self.tfr_by_pulse,
                    audio_length,
                    start_pulse,
                    end_pulse
                )
                
                for shift in shifts:
                    pos_samples.append((file_path, shift))
        
        print(f"  Generated {len(pos_samples)} positive samples")
        return pos_samples, len(pos_samples)
    
    def _generate_neg_samples(self, num_neg_samples):
        """Generate negative samples (without whale pulses)"""
        print(f"\nGenerating negative samples...")
        
        neg_samples = []
        files_to_use = list(self.audio_file_dict.keys())
        
        samples_per_file = num_neg_samples // len(files_to_use)
        remainder = num_neg_samples % len(files_to_use)
        
        for file_idx, filename in enumerate(tqdm(files_to_use, desc="  Processing files")):
            file_path = self.audio_file_dict[filename]
            audio_info = sf.info(file_path)
            audio_length = audio_info.frames
            
            # Skip very short files
            window_size = int(self.window_duration_sec * self.original_sample_rate)
            if audio_length < window_size * 2:
                continue
            
            # Get detections for this file
            file_detections = self.annotations[self.annotations["filename"] == filename]
            detections = list(zip(
                (file_detections["Begin Time (s)"] * self.original_sample_rate).astype(int),
                (file_detections["End Time (s)"] * self.original_sample_rate).astype(int)
            ))
            
            # Generate negative samples for this file
            num_samples_this_file = samples_per_file
            if file_idx < remainder:
                num_samples_this_file += 1
            
            for _ in range(num_samples_this_file):
                start_idx, end_idx = get_random_negative_part(
                    signal_length=audio_length,
                    detections=detections,
                    max_size_selection=window_size,
                    max_try=10
                )
                
                if start_idx is not None:
                    neg_samples.append((file_path, (start_idx, end_idx)))
        
        print(f"  Generated {len(neg_samples)} negative samples")
        return neg_samples, len(neg_samples)
    
    def _generate_shifts(self, shift_amount, signal_length, start_pulse, end_pulse):
        """Generate multiple window positions for a single pulse"""
        shifts = []
        window_size = int(self.window_duration_sec * self.original_sample_rate)
        pulse_len = end_pulse - start_pulse
        
        for _ in range(shift_amount):
            # Random offset for pulse within window
            max_offset = max(0, window_size - pulse_len)
            pulse_offset = torch.randint(0, max_offset + 1, (1,)).item()
            
            start_tfr = start_pulse - pulse_offset
            end_tfr = start_tfr + window_size
            
            # Adjust if out of bounds
            if start_tfr < 0:
                shift = -start_tfr
                start_tfr = 0
                end_tfr += shift
            elif end_tfr > signal_length:
                shift = end_tfr - signal_length
                start_tfr -= shift
                end_tfr = signal_length
            
            shifts.append((start_tfr, end_tfr))
        
        return shifts
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        (file_path, shift), label = self.samples[idx]
        
        # Load audio segment
        audio_np, sr = sf.read(
            file_path,
            start=shift[0],
            frames=shift[1] - shift[0],
            dtype='float32'
        )
        
        # Convert to torch tensor
        audio = torch.from_numpy(audio_np)
        
        # Convert to mono if stereo
        if audio.dim() > 1 and audio.shape[-1] > 1:
            audio = audio.mean(dim=-1)
        
        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        
        # Ensure correct length
        expected_length = int(self.window_duration_sec * self.target_sample_rate)
        if audio.shape[0] < expected_length:
            # Pad
            pad_size = expected_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, pad_size))
        elif audio.shape[0] > expected_length:
            # Trim
            audio = audio[:expected_length]
        
        return {
            'raw_audio': audio,
            'label': torch.tensor(int(label), dtype=torch.long),
            'file_name': os.path.basename(file_path),
        }


def create_supervised_dataloaders(
    data_folder,
    split="val",
    window_duration_sec=8.0,
    sample_rate=16000,
    batch_size=32,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    tfr_by_pulse=5,
    confidence_threshold=None,
    snr_threshold=None,
    seed=42,
):
    """
    Create supervised DataLoader for validation/test
    
    Args:
        data_folder: Path to SEGLVIK root folder
        split: 'val' or 'test'
        window_duration_sec: Window size in seconds
        sample_rate: Target sample rate
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        pin_memory: Whether to pin memory
        tfr_by_pulse: Number of windows per pulse (for positives)
        confidence_threshold: Minimum confidence score for pulses
        snr_threshold: Minimum SNR for pulses
        seed: Random seed
    
    Returns:
        DataLoader with balanced positive/negative samples
    
    Example:
        >>> val_loader = create_supervised_dataloaders(
        ...     data_folder="/path/to/SEGLVIK",
        ...     split="val",
        ...     batch_size=32,
        ...     confidence_threshold=0.5,  # Only use high-confidence detections
        ...     snr_threshold=2.0,         # Only use high-SNR detections
        ... )
    """
    
    dataset = SEGLVIKSupervisedDataset(
        data_folder=data_folder,
        split=split,
        window_duration_sec=window_duration_sec,
        sample_rate=sample_rate,
        tfr_by_pulse=tfr_by_pulse,
        confidence_threshold=confidence_threshold,
        snr_threshold=snr_threshold,
        seed=seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    
    return dataloader
