import torch
from src.utils import *
from src.model import PBT
from src.train import training


def pre_train(config, reduce_num_chs_to):
    test_size = 0.2  # Increased from 0.05 to 0.2 for better validation
    resample = 250
    train_data_set = SeqDataset(
        dim_token=config["d_input"],
        num_tokens_per_channel=config["num_tokens_per_channel"],
        reduce_num_chs_to=reduce_num_chs_to,
        augmentation=config["augmentation"],
    )
    test_data_set = SeqDataset(
        dim_token=config["d_input"],
        num_tokens_per_channel=config["num_tokens_per_channel"],
        reduce_num_chs_to=reduce_num_chs_to,
    )

    # MindBigData EEG dataset (local file)
    print("📊 Loading MindBigData EEG dataset...")
    result = get_mindbigdata_eeg(
        freq_min=config["freq"][0], freq_max=config["freq"][1], resample=resample
    )

    if result is None:
        print("❌ Failed to load MindBigData dataset!")
        print("   Please check:")
        print("   1. Dataset file exists: datasets/EP1.01.txt")
        print("   2. File format is correct")
        print("   3. File is not corrupted")
        print("\n🛑 Training stopped - dataset required!")
        exit(1)

    data, labels, meta, channels = result
    train_data, train_labels, train_meta, test_data, test_labels, test_meta = (
        train_test_split(data, labels, meta, test_size=test_size)
    )

    # Validasi tidak ada overlap antara train dan test menggunakan original indices
    train_original_indices = set(train_meta['original_idx'])
    test_original_indices = set(test_meta['original_idx'])
    assert len(train_original_indices.intersection(test_original_indices)) == 0, "Data leakage detected: overlapping original indices!"

    print(f"Data split validation passed - no overlap between train ({len(train_original_indices)}) and test ({len(test_original_indices)}) sets")
    print(f"Total samples: {len(train_meta) + len(test_meta)} (equals original: {len(meta)})")

    # Normalisasi dilakukan TERPISAH untuk train dan test (mencegah data leakage)
    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)
    train_data_set.append_data_set(
        data_set=train_data, channel_names=channels, label=train_labels
    )
    test_data_set.append_data_set(
        data_set=test_data, channel_names=channels, label=test_labels
    )

    # create data set
    train_data_set.prepare_data_set()
    test_data_set.prepare_data_set(set_pos_channels=train_data_set.dict_channels)

    # Check CUDA availability - STOP if not available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   This model requires GPU training for reasonable performance.")
        print("   Please ensure:")
        print("   1. NVIDIA GPU is installed and drivers are up to date")
        print("   2. CUDA-enabled PyTorch is installed:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Run 'python check_cuda.py' to verify CUDA setup")
        print("\n� Training stopped - CUDA required!")
        exit(1)

    device = torch.device("cuda")

    # Print device information
    print(f"🚀 Using device: {device}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")

    # Enable optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training
    torch.backends.cudnn.allow_tf32 = True
    print("   ✅ CUDA optimizations enabled")

    # Validasi konfigurasi
    assert config["d_model"] % config["num_heads"] == 0
    assert config["d_model"] == 2 * config["d_input"]

    # Hitung num_embeddings dengan fallback
    try:
        num_embeddings = torch.max(
            torch.cat(list(train_data_set.dict_channels.values()))
        ).item() + 1
    except:
        num_embeddings = 64

    model = PBT(
        d_input=config["d_input"],
        n_classes=10,
        num_embeddings=num_embeddings,
        num_tokens_per_channel=config["num_tokens_per_channel"],
        d_model=config["d_model"],
        n_blocks=config["num_transformer_blocks"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        device=device,
        learnable_cls=config["learnable_cls"],
        bias_transformer=config["bias_transformer"],
        bert=False#True if config["bert_supervised"] or config["pre_train_bert"] else False,
    )

    # 3. Pindahkan model ke device
    model = model.to(device)
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PBT Model initialized for MNIST classification")
    print(f"Parameters: {total_params:,}")
    print(f"Classes: 10 (MNIST digits)")

    training(
        parameter=config,
        model=model,
        train_data_set=train_data_set,
        test_data_set=test_data_set,
        n_classes=10,  # MNIST has 10 classes (digits 0-9)
        num_workers=4,
    )


if __name__ == "__main__":
    config = {
        "pre_train_bert": False,  # unsupervised pre-training in BERT-style
        # if false, supervised pre-training
        # Pre - Processing
        #"freq": [8, 45],
        "freq": [0.5, 40],  # atau [0.5, 40] tergantung karakteristik data visual
        "normalization": "zscore",
        # Model (Reduced complexity to prevent overfitting)
        "d_input": 32,
        "d_model": 64,  # Input gets expanded in lin. projection
        "dim_feedforward": 64 * 2,  # Reduced from 64*4 to 64*2
        "num_tokens_per_channel": 7,  # For 256 samples EPOC (7*32=224 < 256)
        "num_transformer_blocks": 2,  # Reduced from 4 to 2
        "num_heads": 4,  # Reduced from 8 to 4
        "bert_supervised": False,  # add a reconstruction task (BERT) as regularisation to loss
        "learnable_cls": False,
        "bias_transformer": True,
        # Train Hyper-Parameters (Optimized for larger dataset)
        "lr": 1e-4,  # Increased from 5e-5 for larger dataset
        "lr_warm_up_iters": 100,  # Increased from 50 for more gradual warmup
        "batch_size": 64,  # Increased from 32 for larger dataset efficiency
        "num_epochs": 300,  # Increased from 200 for larger dataset
        "betas": (0.9, 0.95),  # betas AdamW
        "clip_gradient": 1.0,
        # Regularization & Augmentation (Enhanced for overfitting prevention)
        "weight_decay": 0.05,  # Increased from 0.01 to 0.05
        "weight_decay_pos_embedding": 0.01,  # Increased from 0.0 to 0.01
        "weight_decay_cls_head": 1,  # cls_head = classification head (linear layer)
        # higher for pre-train may improve few-shot adaptation
        "dropout": 0.2,  # Increased from 0.1 to 0.2
        "label_smoothing": 0.15,  # Increased from 0.1 to 0.15
        # Augmentasi yang didukung: time_shifts, DC_shifts, amplitude_scaling, noise
        "augmentation": ["time_shifts", "noise", "amplitude_scaling"],  # Added amplitude_scaling
        # WandB
        "wandb_log": False,  # Disabled for testing
        "wandb_name": False,
        "wandb_proj": "PatchedBrainTransformer",
        "wandb_watch": True,
        "save": "models/mnist_brain_transformer",  # Path untuk menyimpan model
        "checkpoints": 20,
        "load": False,
        "seed": 42,  # set random seed
        "compile_model": False,  # compile model with PyTroch to speed up
        "plot_training": True,  # Enable real-time plotting of training curves
        "mixed_precision": True,  # Enable Automatic Mixed Precision (AMP) for faster training
        # Early stopping to prevent overfitting
        "early_stopping": True,
        "early_stopping_patience": 100,  # Stop if no improvement for 100 epochs (increased for larger dataset)
        "early_stopping_min_delta": 0.001,  # Minimum change to qualify as improvement
    }

    # Opsi 1: Single run dengan seed 42 (untuk reproduksibilitas penuh)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    pre_train(config, reduce_num_chs_to=30)

    # Opsi 2: Multiple runs dengan seed deterministik (uncomment jika diperlukan)
    # seeds = [42, 123, 456]  # seed yang tetap untuk reproduksibilitas
    # for seed in seeds:
    #     config["seed"] = seed
    #     torch.manual_seed(config["seed"])
    #     torch.cuda.manual_seed(config["seed"])
    #     np.random.seed(config["seed"])
    #     random.seed(config["seed"])
    #     torch.backends.cudnn.deterministic = True
    #
    #     print(f"Running with seed: {seed}")
    #     pre_train(config, reduce_num_chs_to=30)
