import os
from flask import Flask, request, jsonify
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the fine-tuned BERT model and tokenizer
output_dir = './results'
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

departments = {
    0: 'data',
    1: 'infra',
    2: 'engineering'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'new_texts' not in data:
        return jsonify({'error': 'new_texts field is required'}), 400
    
    new_texts = data['new_texts']
    
    # Tokenize and predict
    inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

    results = [
        {
            'text': text,
            'predicted_label': pred.item(),
            'predicted_department': departments[pred.item()]
        }
        for text, pred in zip(new_texts, predictions)
    ]
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
