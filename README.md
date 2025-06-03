# 🔤 Next Word Prediction Using Recurrent Neural Networks (Language Modeling)

## 📌 Project Overview

This project builds a word-level **language model** using RNN architectures (e.g., LSTM, GRU) to **predict the next word** in a sequence. It demonstrates how deep learning handles sequential text data and is a foundational NLP task for applications like text autocompletion, chatbots, and machine translation.

---

## 🎯 Skills Takeaway

- Deep Learning for Sequential Data  
- Natural Language Processing (NLP) Fundamentals  
- RNN, LSTM, GRU Architectures  
- Text Preprocessing and Tokenization  
- Language Modeling and Text Generation  
- Model Deployment Basics  

---

## 🧠 Domain

**Deep Learning – Natural Language Processing**

---

## ❓ Problem Statement

The objective is to create a neural language model that predicts the **next word** in a sentence based on the previous context using **RNN-based models** like LSTM or GRU. This helps learners understand temporal data dependencies and token-based prediction.

---

## 💼 Business Use Cases

- **Text Editors**: Suggest context-aware words while typing.
- **Voice Assistants**: Improve speech-to-text predictions.
- **Search Engines**: Autocomplete search queries.
- **Chatbots & Translators**: Used in encoder-decoder models for generating responses or translations.

---

## 🧭 Approach

1. **Load Dataset**: Use `WikiText-2` from Hugging Face Datasets.
2. **Text Preprocessing**:
   - Tokenization
   - Lowercasing
   - Vocabulary creation
   - Padding/truncating
3. **Create Sequences**:
   - Generate input sequences and next-word targets.
4. **Model Development**:
   - Build an RNN/LSTM-based model using TensorFlow or PyTorch.
5. **Train the Model**:
   - Predict next word using cross-entropy loss.
6. **Evaluate**:
   - Plot training/validation loss
   - Measure prediction accuracy
7. **Generate Predictions**:
   - Produce text by generating next words from a seed sequence.

---

## ✅ Results

- A trained **word-level language model** that can generate realistic text continuations.
- Clear visualizations of training performance.
- Sample predictions showing real use-case behavior.

---

## 📈 Evaluation Metrics

- **Accuracy** – Ratio of correct next-word predictions.
- **Loss** – Use CrossEntropyLoss (PyTorch) or sparse categorical crossentropy (TensorFlow).
- **Loss Curves** – Visualize learning behavior across epochs.

---

## 🧾 Technical Tags

`NLP`, `RNN`, `LSTM`, `GRU`, `Language Modeling`, `Text Generation`, `TensorFlow`, `PyTorch`, `Deep Learning`, `Hugging Face`, `Next Word Prediction`

---

## 📊 Dataset

- **Name**: WikiText-2
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/wikitext)
- **Format**: Plain text files
- **Structure**: Raw Wikipedia articles split into word-level tokens.

---

## 📁 Dataset Explanation

- Clean and structured Wikipedia articles.
- Rich in sentence variety, punctuation, and grammar.
- Suitable for real-world NLP model training.

### 🧼 Preprocessing Includes:

- Lowercasing and stripping punctuation (optional).
- Tokenization and vocabulary creation.
- Padding sequences to a uniform length.
- Splitting into train and validation sets.

---

## 📦 Project Deliverables

- 🧾 **Source Code**: Python scripts or notebooks for preprocessing, model creation, and training.
- 🧠 **Trained Model**: Saved `.h5` or `.pt` file.
- 📊 **Training Results**: Loss curves and example predictions.
- 🚀 ** Streamlit or CLI text generation app.

