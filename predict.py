import torch
import os
from config import GPT_CONFIG_124M
from model import GPTModel
from utils import get_tokenizer
from main import modify_model_for_classification
from classify import classify_review

def load_trained_model(model_path="spam_classifier_model.pth", device=None):
    """
    Loads the trained model architecture and weights.
    """
    if device is None:
         # --- UPDATED DEVICE SELECTION FOR MAC AMD (MPS) ---
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
    print(f"Using device: {device}")

    # 1. Initialize architecture
    model = GPTModel(GPT_CONFIG_124M)
    
    # 2. Modify for classification (change output head)
    # Important: This must be done BEFORE loading state_dict if the state_dict contains the modified head
    model = modify_model_for_classification(model)
    
    # 3. Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
        
    print(f"Loading model from {model_path}...")
    map_location = device if device.type == 'cpu' else None # Handle mapping if moving from GPU to CPU
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    return model, device

def main():
    print("=== Spam Classifier Prediction ===")
    
    # Load model and tokenizer
    try:
        model, device = load_trained_model()
        tokenizer = get_tokenizer()
        
        # Load max_length from metadata
        if os.path.exists("model_metadata.txt"):
            with open("model_metadata.txt", "r") as f:
                max_length = int(f.read().strip())
            print(f"Loaded max_length from metadata: {max_length}")
        else:
            # # Fallback if file doesn't exist (e.g. before retraining)
            # max_length = 120 # Based on last run
            print(f"Warning: model_metadata.txt not found. Using fallback max_length: {max_length}")
            
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    print("\nType a message to check if it's spam. Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        user_input = input("\nEnter message: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input.strip():
            continue
            
        result = classify_review(user_input, model, tokenizer, device, max_length=max_length)
        
        # Add some color/formatting to output
        if result == "spam":
            print(f"Result: 🔴 SPAM")
        else:
            print(f"Result: 🟢 NOT SPAM")

if __name__ == "__main__":
    main()
