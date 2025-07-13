# Machine Translator using Machine Learning

## Overview
This project is a web-based machine translation application that leverages state-of-the-art transformer models to translate text between multiple languages. It uses the Hugging Face Transformers library and pre-trained models for high-quality translation. The backend is built with Flask, and the frontend is a simple HTML interface.

## Features
- Translate text between English, Hindi, and French (extendable to more languages)
- Uses pre-trained transformer models from Hugging Face
- Web interface for easy interaction
- Extensible for training and testing custom models

## Supported Language Pairs
- English ↔ Hindi
- English ↔ French
- (Easily extendable by adding more models in `app.py`)

## Project Structure
```
machine translator/
├── doc/                # Project reports and documentation
├── project/
│   ├── app.py          # Main Flask application
│   ├── templates/
│   │   └── index.html  # Frontend HTML page
│   └── notebooks/
│       └── En_Hi_language_translation.ipynb  # Jupyter notebook for model exploration
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "machine translator"
   ```
2. **Install dependencies:**
   ```bash
   pip install flask torch transformers datasets
   ```
3. **Run the application:**
   ```bash
   python project/app.py
   ```
4. **Access the web app:**
   Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## Usage
- Enter the text you want to translate in the web interface.
- Select the source and target languages.
- Click the translate button to see the result.

## Custom Model Training (Optional)
- The code includes functions to train and test your own translation models using the IIT Bombay English-Hindi dataset.
- Uncomment and run `train_model()` or `test_model()` in `app.py` as needed.
- Trained models are saved locally and can be loaded for inference.

## Extending to More Languages
- Add new language pairs and their Hugging Face model checkpoints to the `LANGUAGE_MODELS` dictionary in `app.py`.
- Restart the Flask server to enable new language pairs.

## Requirements
- Python 3.8+
- Flask
- torch
- transformers
- datasets

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes
4. Push to your branch (`git push origin feature-branch`)
5. Open a Pull Request

## License
This project is for educational purposes. See the `doc/` folder for research and project reports.

## Acknowledgements
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [IIT Bombay English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
- [Helsinki-NLP OPUS-MT Models](https://huggingface.co/Helsinki-NLP) 