"""
Main entry point for the Spam Classifier with Pretrained GPT-2 Weights.
This script orchestrates the complete pipeline:
1. Download GPT-2 pretrained weights from OpenAI
2. Download and prepare spam data
3. Create datasets and dataloaders
4. Initialize model with pretrained weights
5. Run training (fine-tuning)
6. Evaluate on test set
7. Demonstrate classification
"""

import time
import torch

from config import GPT_CONFIG_124M
from model import GPTModel
from dataset import create_dataloaders
from train import train_classifier_simple
from evaluate import calc_accuracy_loader
from classify import classify_review
from utils import download_and_prepare_data, get_tokenizer
from gpt_download import download_and_load_gpt2
from weights import load_weights_into_gpt


def modify_model_for_classification(model, num_classes=2):
    """
    Modify the GPT model for classification by replacing the output head.
    Uses only the last token's output for classification.
    """
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace output head with classification head
    # We use manual_seed to ensure reproducibility of the new random weights
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)
    
    # Make the new output head trainable
    for param in model.out_head.parameters():
        param.requires_grad = True
    
    # Also make last transformer block and final layer norm trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    return model


def main():
    # --- UPDATED DEVICE SELECTION FOR MAC AMD (MPS) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Mac AMD/Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    # --------------------------------------------------
    
    # Step 1: Download pretrained GPT-2 weights
    print("\n=== Step 1: Downloading Pretrained GPT-2 Weights ===")
    model_size = "124M"
    models_dir = "gpt2_weights"
    # This function uses TensorFlow to read the original OpenAI checkpoint
    settings, params = download_and_load_gpt2(model_size, models_dir)
    print(f"GPT-2 {model_size} weights loaded successfully!")
    print(f"Settings: n_layer={settings['n_layer']}, n_head={settings['n_head']}, n_embd={settings['n_embd']}")
    
    # Step 2: Download and prepare spam data
    print("\n=== Step 2: Preparing Spam Data ===")
    data_path = download_and_prepare_data(data_dir="data")
    
    # Step 3: Initialize tokenizer and create dataloaders
    print("\n=== Step 3: Creating Dataloaders ===")
    tokenizer = get_tokenizer()
    
    train_loader, val_loader, test_loader, max_length = create_dataloaders(
        train_csv=data_path / "train.csv",
        val_csv=data_path / "validation.csv",
        test_csv=data_path / "test.csv",
        tokenizer=tokenizer,
        batch_size=8
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Step 4: Initialize model with pretrained weights
    print("\n=== Step 4: Initializing Model with Pretrained Weights ===")
    # Initialize the architecture with the configuration from config.py
    model = GPTModel(GPT_CONFIG_124M)
    
    # Load the pretrained weights (transfers numpy arrays to pytorch params)
    load_weights_into_gpt(model, params)
    print("Pretrained weights loaded into model!")
    
    # Modify the model structure for spam classification (binary output)
    model = modify_model_for_classification(model)
    
    # Move model to the selected device (GPU/CPU)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 5: Training (Fine-tuning)
    print("\n=== Step 5: Fine-tuning on Spam Data ===")
    start_time = time.time()
    
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        calc_accuracy_loader_fn=calc_accuracy_loader
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\nTraining completed in {execution_time_minutes:.2f} minutes.")
    
    # Step 6: Final Evaluation
    print("\n=== Step 6: Final Evaluation ===")
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Step 7: Demo classification
    print("\n=== Step 7: Demo Classification ===")
    test_messages = [
        "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
        "Hey, are you coming to the party tonight?",
        "URGENT! Your mobile number has won a $2000 prize. Call now!",
        "Can you pick up some groceries on your way home?",
    ]
    
    for text in test_messages:
        result = classify_review(text, model, tokenizer, device, max_length=max_length)
        print(f"\nText: {text[:50]}...")
        print(f"Classification: {result}")
    
    # Step 8: Save model and metadata
    print("\n=== Step 8: Saving Model and Metadata ===")
    torch.save(model.state_dict(), "spam_classifier_model.pth")
    print("Model saved to spam_classifier_model.pth")
    
    # Save max_length for prediction
    with open("model_metadata.txt", "w") as f:
        f.write(str(max_length))
    print(f"Max length ({max_length}) saved to model_metadata.txt")


if __name__ == "__main__":
    main()
