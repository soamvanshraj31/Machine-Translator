from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import threading
import os
from datasets import load_dataset

# Initialize Flask app
app = Flask(__name__)

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Supported language pairs and their model checkpoints
LANGUAGE_MODELS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    # Add more language pairs here
}

# Cache for loaded models and tokenizers
model_cache = {}
cache_lock = threading.Lock()

def get_model_and_tokenizer(from_lang, to_lang):
    key = (from_lang, to_lang)
    if key not in LANGUAGE_MODELS:
        return None, None
    with cache_lock:
        if key not in model_cache:
            checkpoint = LANGUAGE_MODELS[key]
            try:
                logging.info(f"Loading model and tokenizer for {from_lang}->{to_lang}: {checkpoint}")
                tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
                model_cache[key] = (tokenizer, model)
            except Exception as e:
                logging.error(f"Failed to load model {checkpoint}: {e}")
                return None, None
        return model_cache[key]

# =====================
# Model Training Section
# =====================
def train_model():
    """
    Train a translation model (example: English-Hindi) and save it locally.
    Only run this when you want to retrain the model.
    """
    model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
    raw_datasets = load_dataset("cfilt/iitb-english-hindi")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    max_input_length = 128
    max_target_length = 128
    source_lang = "en"
    target_lang = "hi"

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_train_epochs = 1
    from transformers import DataCollatorForSeq2Seq, AdamWeightDecay
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    train_dataset = model.prepare_tf_dataset(
        tokenized_datasets["test"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    validation_dataset = model.prepare_tf_dataset(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
    model.compile(optimizer=optimizer)
    model.fit(train_dataset, validation_data=validation_dataset, epochs=num_train_epochs)
    model.save_pretrained("tf_model/")
    print("Model trained and saved to tf_model/")

# =====================
# Model Testing Section
# =====================
def test_model():
    """
    Test the trained model on a few examples.
    """
    model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained("tf_model/")
    test_data = [
        {"en": "what are you doing here", "hi": "तुम यहाँ क्या कर रहे हो"},
        {"en": "I am going to school", "hi": "मैं स्कूल जा रहा हूँ"},
        {"en": "I like this", "hi": "मुझे यह पसंद है"}
    ]
    y_true = [example["hi"] for example in test_data]
    y_pred = []
    for example in test_data:
        input_text = example["en"]
        tokenized = tokenizer([input_text], return_tensors='pt')
        out = model.generate(**tokenized, max_length=128)
        prediction = tokenizer.decode(out[0], skip_special_tokens=True)
        y_pred.append(prediction)
    exact_matches = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
    accuracy = (exact_matches / len(y_true)) * 100
    print("Ground Truth:", y_true)
    print("Predictions:", y_pred)
    print(f"Exact Match Accuracy: {accuracy:.2f}%")

# =====================
# Flask Inference Section
# =====================
@app.route('/')
def index():
    # Render the HTML file
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data.get("english_text", "")  # For now, keep the frontend param
        from_lang = data.get("from_lang", "en")
        to_lang = data.get("to_lang", "hi")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        tokenizer, model = get_model_and_tokenizer(from_lang, to_lang)
        if not tokenizer or not model:
            return jsonify({"error": f"Translation from {from_lang} to {to_lang} not supported or model could not be loaded."}), 400

        logging.debug(f"Translating ({from_lang}->{to_lang}): {text}")
        inputs = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"translation": translation})
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Uncomment to train or test the model as needed:
    # train_model()
    # test_model()
    app.run(debug=True)

# =====================
# Requirements:
# pip install flask torch transformers datasets
# =====================
