import warnings
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
# Import AMP components with proper fallback for PyTorch versions
try:
    # Try PyTorch 1.6+ style first (more common)
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
    print("🔧 Using torch.cuda.amp (PyTorch 1.6+ style)")
except ImportError:
    try:
        # Try PyTorch 2.0+ style
        from torch.amp import autocast, GradScaler
        AMP_AVAILABLE = True
        print("🔧 Using torch.amp (PyTorch 2.0+ style)")
    except ImportError:
        # No AMP available
        autocast = None
        GradScaler = None
        AMP_AVAILABLE = False
        print("⚠️ AMP not available - using standard precision")
import matplotlib.pyplot as plt
import numpy as np

from src.model import LearningRateScheduler


def plot_training_progress(epochs, train_losses, test_losses, train_accs, test_accs, save_path="plots/"):
    """Plot training progress in real-time"""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Overfitting indicator
    plt.subplot(1, 3, 3)
    if len(train_losses) > 0 and len(test_losses) > 0:
        loss_diff = np.array(test_losses) - np.array(train_losses)
        plt.plot(epochs, loss_diff, 'g-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss - Train Loss')
        plt.title('Overfitting Indicator')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close to prevent memory issues


def training(
    parameter,
    model,
    train_data_set,
    test_data_set,
    n_classes,
    num_workers=1,
    chose_optimizer="customAdamW",
):

    # Ensure CUDA is available for training
    if not torch.cuda.is_available():
        print("❌ CUDA is not available for training!")
        print("   Training requires GPU. Please check CUDA setup.")
        raise RuntimeError("CUDA required for training - use 'python check_cuda.py' to verify setup")

    device = torch.device("cuda")
    model.to(device)

    # Print device and memory info
    print(f"🚀 Training on device: {device}")
    print(f"   GPU Memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    print(f"   GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    # Clear cache to start fresh
    torch.cuda.empty_cache()
    print("   ✅ GPU cache cleared")

    # log with WandB
    if parameter["wandb_log"]:
        import wandb

        # get rid of parameter that should not be logged
        wandb_dict = {
            k: parameter[k]
            for k in parameter.keys()
            - ["wandb_log", "compile_model", "wandb_proj", "data_set", "wandb_name"]
        }
        if parameter["wandb_name"]:
            wandb.init(
                project=parameter["wandb_proj"],
                config=wandb_dict,
                reinit=False,
                name=parameter["wandb_name"],
            )
        else:
            wandb.init(project=parameter["wandb_proj"], config=wandb_dict, reinit=False)

        if parameter["wandb_watch"]:
            wandb.watch(models=model, log="all", log_freq=25)

    criterion = nn.CrossEntropyLoss(label_smoothing=parameter["label_smoothing"]).to(
        device
    )

    if chose_optimizer == "customAdamW":
        optimizer = model.configure_optimizers(
            weight_decay=parameter["weight_decay"],
            weight_decay_cls_head=parameter["weight_decay_cls_head"],
            learning_rate=parameter["lr"],
            betas=parameter["betas"],
            device_type=device,
        )

    elif chose_optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=parameter["lr"],
            betas=(0.9, 0.95),
            weight_decay=parameter["weight_decay"],
        )
    else:
        raise ValueError(f"Optimizer {chose_optimizer} is not supported")

    learning_rate_scheduler = LearningRateScheduler(
        warmup_iters=parameter["lr_warm_up_iters"],
        learning_rate=parameter["lr"],
        lr_decay_iters=parameter["num_epochs"],
        min_lr=parameter["lr"] * 1 / 10,
    )

    # compile_model
    if parameter["compile_model"]:
        # torch.compile requires PyTorch >=2.0
        if int(torch.__version__[0]) >= 2:
            model = torch.compile(model)
            print("Model is compiled")
        else:
            warnings.warn(
                "Compile model requires PyTorch >= 2.0, model is NOT compiled!"
            )

    # --------------------------------------------------------------------------------------------------------------
    # Training Loop
    if not train_data_set.dict_channels == test_data_set.dict_channels:
        raise ValueError("Channel index between train and test is not consistence")

    train_generator = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=parameter["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=train_data_set.my_collate,
        num_workers=num_workers,
    )

    test_generator = torch.utils.data.DataLoader(
        test_data_set,
        batch_size=parameter["batch_size"],
        shuffle=False,
        drop_last=False,
        collate_fn=test_data_set.my_collate,
        num_workers=num_workers,
    )

    # Initialize lists for plotting
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # Early stopping variables
    best_test_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = parameter.get("early_stopping_patience", 20)
    early_stopping_min_delta = parameter.get("early_stopping_min_delta", 0.001)
    early_stopping_enabled = parameter.get("early_stopping", False)

    # Initialize AMP scaler if mixed precision is enabled and available
    use_amp = (parameter.get("mixed_precision", False) and
               device.type == 'cuda' and
               AMP_AVAILABLE)

    if use_amp:
        scaler = GradScaler()  # torch.cuda.amp.GradScaler() doesn't need device parameter
    else:
        scaler = None

    if use_amp:
        print("⚡ Mixed Precision Training (AMP) enabled")
    elif parameter.get("mixed_precision", False):
        print("⚠️ Mixed Precision requested but not available (using standard precision)")
    else:
        print("🔧 Standard precision training")

    for ml_epochs in range(parameter["num_epochs"]):
        lr = learning_rate_scheduler.get_lr(iteration=ml_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # save loss and labels within an epoch to log the average across the epoch
        current_train_loss, current_test_loss = 0, 0
        current_train_loss_bert, current_test_loss_bert = 0, 0
        true_labels_train = torch.empty(0, dtype=torch.float).to(device)
        pred_labels_train = torch.empty(0, dtype=torch.float).to(device)
        true_labels_test = torch.empty(0, dtype=torch.float).to(device)
        pred_labels_test = torch.empty(0, dtype=torch.float).to(device)

        # --------------------------------------------------------------------------------------------------------------
        for i, data in enumerate(train_generator):
            num_trials = sum([x.size(0) for x in data["patched_eeg_token"]])
            model.train()
            optimizer.zero_grad()

            logits = torch.empty(0, n_classes).to(device)
            target = torch.empty(0)
            current_help_loss = 0

            # loop over mini batches with same number of tokens
            for sub_batch in range(len(data["patched_eeg_token"])):
                # Use autocast for mixed precision
                if use_amp:
                    autocast_context = autocast(enabled=True)
                else:
                    # No-op context manager for non-AMP training
                    from contextlib import nullcontext
                    autocast_context = nullcontext()

                with autocast_context:
                    transformer_out, logits1, pos_masking = model.forward(
                        x=data["patched_eeg_token"][sub_batch].to(device),
                        pos=data["pos_as_int"][sub_batch].type(torch.LongTensor).to(device),
                    )

                if parameter["pre_train_bert"]:
                    if True in pos_masking:
                        loss = model.cos_sim_loss(
                            output=data["patched_eeg_token"][sub_batch][:, 1:].to(
                                device
                            )[pos_masking],
                            target=transformer_out[pos_masking],
                        ) * (data["labels"][sub_batch].size(0) / num_trials)

                        # Backward pass with AMP
                        if use_amp:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        logits = torch.cat((logits, logits1), dim=0)
                        target = torch.cat((target, data["labels"][sub_batch]), dim=0)
                        current_help_loss += loss

                else:
                    # scale loss depending of the number of train-trials
                    loss = criterion(
                        logits1,
                        data["labels"][sub_batch].type(torch.LongTensor).to(device),
                    ) * (data["labels"][sub_batch].size(0) / num_trials)

                    # Backward pass with AMP
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    logits = torch.cat((logits, logits1), dim=0)
                    target = torch.cat((target, data["labels"][sub_batch]), dim=0)
                    current_help_loss += loss

            current_train_loss += current_help_loss

            # Gradient clipping and optimizer step with AMP
            if use_amp:
                if parameter["clip_gradient"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=parameter["clip_gradient"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                if parameter["clip_gradient"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=parameter["clip_gradient"]
                    )
                optimizer.step()

            pred_labels_train = torch.cat((pred_labels_train, logits.argmax(dim=1)), 0)
            true_labels_train = torch.cat((true_labels_train, target.to(device)), 0)

        # --------------------------------------------------------------------------------------------------------------
        # Evaluation step
        for j, data in enumerate(test_generator):
            num_trials = sum([x.size(0) for x in data["patched_eeg_token"]])
            with torch.no_grad():
                model.eval()

                logits = torch.empty(0, n_classes).to(device)
                target = torch.empty(0)
                current_help_loss = 0

                for sub_batch in range(len(data["patched_eeg_token"])):
                    transformer_out, logits1, pos_masking = model.forward(
                        x=data["patched_eeg_token"][sub_batch].to(device),
                        pos=data["pos_as_int"][sub_batch]
                        .type(torch.LongTensor)
                        .to(device),
                    )

                    if parameter["pre_train_bert"]:
                        if True in pos_masking:
                            loss = model.cos_sim_loss(
                                output=data["patched_eeg_token"][sub_batch][:, 1:].to(
                                    device
                                )[pos_masking],
                                target=transformer_out[pos_masking],
                            ) * (data["labels"][sub_batch].size(0) / num_trials)

                            logits = torch.cat((logits, logits1), dim=0)
                            target = torch.cat(
                                (target, data["labels"][sub_batch]), dim=0
                            )
                            current_help_loss += loss
                    else:
                        # scale loss depending of the number of train-trials
                        loss = criterion(
                            logits1,
                            data["labels"][sub_batch].type(torch.LongTensor).to(device),
                        ) * (data["labels"][sub_batch].size(0) / num_trials)

                        logits = torch.cat((logits, logits1), dim=0)
                        target = torch.cat((target, data["labels"][sub_batch]), dim=0)
                        current_help_loss += loss

                current_test_loss += current_help_loss
                pred_labels_test = torch.cat(
                    (pred_labels_test, logits.argmax(dim=1)), 0
                )
                true_labels_test = torch.cat((true_labels_test, target.to(device)), 0)

        current_train_loss = current_train_loss.cpu().detach().numpy() / (i + 1)
        current_test_loss = current_test_loss.cpu().detach().numpy() / (j + 1)

        if parameter["bert_supervised"]:
            current_train_loss_bert = current_train_loss_bert.cpu().detach().numpy() / (
                i + 1
            )
            current_test_loss_bert = current_test_loss_bert.cpu().detach().numpy() / (
                j + 1
            )

        if not parameter["pre_train_bert"]:
            current_acc_train = torch.sum(
                true_labels_train == pred_labels_train
            ) / true_labels_train.size(0)
            current_acc_test = torch.sum(
                true_labels_test == pred_labels_test
            ) / true_labels_test.size(0)

        if parameter["wandb_log"]:
            if parameter["bert_supervised"]:
                wandb.log(
                    {
                        "train_loss": current_train_loss,
                        "test_loss": current_test_loss,
                        "train_loss_BERT": current_train_loss_bert,
                        "test_loss_BERT": current_test_loss_bert,
                        "acc_train": current_acc_train,
                        "acc_test": current_acc_test,
                    }
                )
            elif parameter["pre_train_bert"]:
                wandb.log(
                    {
                        "train_loss": current_train_loss,
                        "test_loss": current_test_loss,
                    }
                )
            else:
                wandb.log(
                    {
                        "train_loss": current_train_loss,
                        "test_loss": current_test_loss,
                        "acc_train": current_acc_train,
                        "acc_test": current_acc_test,
                    }
                )

        else:
            print(
                "\n Epoch: {}, train_loss {}, test_loss {}".format(
                    ml_epochs, current_train_loss, current_test_loss
                )
            )
            if not parameter["pre_train_bert"]:
                print(
                    "Epoch: {}, train_acc {}, test_acc {}".format(
                        ml_epochs, current_acc_train, current_acc_test
                    )
                )
                if parameter["bert_supervised"]:
                    print(
                        "Epoch: {}, train_loss_BERT {}, test_loss_BERT {}".format(
                            ml_epochs, current_acc_train, current_acc_test
                        )
                    )

        # Store metrics for plotting
        epoch_list.append(ml_epochs)
        train_loss_list.append(current_train_loss)
        test_loss_list.append(current_test_loss)

        if not parameter["pre_train_bert"]:
            train_acc_list.append(current_acc_train.cpu().numpy() if torch.is_tensor(current_acc_train) else current_acc_train)
            test_acc_list.append(current_acc_test.cpu().numpy() if torch.is_tensor(current_acc_test) else current_acc_test)
        else:
            train_acc_list.append(0)  # No accuracy for BERT pre-training
            test_acc_list.append(0)

        # Early stopping check
        if early_stopping_enabled:
            if current_test_loss < best_test_loss - early_stopping_min_delta:
                best_test_loss = current_test_loss
                patience_counter = 0
                print(f"   ✅ New best test loss: {best_test_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement for {patience_counter}/{early_stopping_patience} epochs")

                if patience_counter >= early_stopping_patience:
                    print(f"\n🛑 Early stopping triggered after {ml_epochs + 1} epochs")
                    print(f"   Best test loss: {best_test_loss:.4f}")
                    break

        # GPU Memory monitoring (every 50 epochs)
        if device.type == 'cuda' and (ml_epochs + 1) % 50 == 0:
            print(f"   📊 Epoch {ml_epochs+1} - GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

        # Plot every 10 epochs or at the end (if plotting is enabled)
        if parameter.get("plot_training", False) and ((ml_epochs + 1) % 10 == 0 or ml_epochs == parameter["num_epochs"] - 1):
            if parameter["save"]:
                plot_path = parameter["save"] if not parameter["wandb_log"] else os.path.join(parameter["save"], wandb.run.name if parameter["wandb_log"] else "")
                plot_training_progress(epoch_list, train_loss_list, test_loss_list, train_acc_list, test_acc_list, plot_path)
            else:
                plot_training_progress(epoch_list, train_loss_list, test_loss_list, train_acc_list, test_acc_list)

        if parameter["checkpoints"]:
            if parameter["save"]:
                if parameter["wandb_log"]:
                    path = os.path.join(parameter["save"], wandb.run.name)
                else:
                    path = parameter["save"]
                if not os.path.isdir(path):
                    os.mkdir(path)

                if ml_epochs % parameter["checkpoints"] == 0 and ml_epochs != 0:
                    if not parameter["pre_train_bert"]:
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "learning_rate_scheduler": learning_rate_scheduler.state_dict(),
                                "config": parameter,
                                "epoch": ml_epochs,
                                "train_loss": current_train_loss,
                                "test_loss": current_test_loss,
                                "acc_train": current_acc_train,
                                "acc_test": current_acc_test,
                                "dict_channels": train_data_set.dict_channels,
                            },
                            os.path.join(path, "checkpoint_" + str(ml_epochs) + ".pt"),
                        )
                    else:
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "learning_rate_scheduler": learning_rate_scheduler.state_dict(),
                                "config": parameter,
                                "epoch": ml_epochs,
                                "train_loss": current_train_loss,
                                "test_loss": current_test_loss,
                                "dict_channels": train_data_set.dict_channels,
                            },
                            os.path.join(path, "checkpoint_" + str(ml_epochs) + ".pt"),
                        )
            else:
                warnings.warn("Checkpoints are not saved!")

    if parameter["save"]:
        if parameter["wandb_log"]:
            path = os.path.join(parameter["save"], wandb.run.name)
        else:
            path = parameter["save"]
        if not os.path.isdir(path):
            os.mkdir(path)

        if not parameter["pre_train_bert"]:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "learning_rate_scheduler": learning_rate_scheduler.state_dict(),
                    "config": parameter,
                    "epoch": ml_epochs,
                    "train_loss": current_train_loss,
                    "test_loss": current_test_loss,
                    "acc_train": current_acc_train,
                    "acc_test": current_acc_test,
                    "dict_channels": train_data_set.dict_channels,
                },
                os.path.join(path, "final_" + str(ml_epochs) + ".pt"),
            )
        else:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "learning_rate_scheduler": learning_rate_scheduler.state_dict(),
                    "config": parameter,
                    "epoch": ml_epochs,
                    "train_loss": current_train_loss,
                    "test_loss": current_test_loss,
                    "dict_channels": train_data_set.dict_channels,
                },
                os.path.join(path, "final_" + str(ml_epochs) + ".pt"),
            )

    if parameter["wandb_log"]:
        wandb.finish()
