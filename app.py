from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import os
from config import GPT_CONFIG_124M
from model import GPTModel
from utils import get_tokenizer
from main import modify_model_for_classification
from classify import classify_review

app = Flask(__name__)
CORS(app)

# Global variables for model and metadata
model = None
tokenizer = None
device = None
max_length = 120  # Default fallback

def load_model_for_web():
    global model, tokenizer, device, max_length
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Initialize and load model
    model = GPTModel(GPT_CONFIG_124M)
    model = modify_model_for_classification(model)
    
    model_path = "spam_classifier_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return False
        
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = get_tokenizer()
    
    # Load max_length metadata
    if os.path.exists("model_metadata.txt"):
        with open("model_metadata.txt", "r") as f:
            max_length = int(f.read().strip())
            
    print(f"Web backend initialized with max_length: {max_length}")
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
        
    # Perform classification
    result = classify_review(text, model, tokenizer, device, max_length=max_length)
    
    return jsonify({
        'text': text,
        'prediction': result,
        'is_spam': result == 'spam'
    })

if __name__ == '__main__':
    if load_model_for_web():
        app.run(debug=True, port=5001)
    else:
        print("Failed to load model. Please ensure 'spam_classifier_model.pth' exists.")
