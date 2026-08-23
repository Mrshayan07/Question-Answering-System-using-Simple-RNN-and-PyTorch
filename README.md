# Question Answering System using Simple RNN

A simple **Question Answering (QA) system** built from scratch using **PyTorch**. The project processes a custom question-answer dataset, builds a vocabulary, converts text into numerical representations, and uses an **Embedding + Simple RNN** architecture to predict an answer token for a given question.

## 🚀 Project Overview

The project demonstrates the basic workflow of building a neural network for a text-based task:

* Load a custom Question-Answer dataset
* Tokenize questions and answers
* Build a vocabulary
* Convert text into numerical indices
* Handle unknown words using an `<UNK>` token
* Create a custom PyTorch `Dataset`
* Load data using `DataLoader`
* Convert words into embeddings
* Process question sequences using a Simple RNN
* Train the model using Cross-Entropy Loss
* Generate predictions for new questions

## 🧠 Model Architecture

```text id="qa9x1r"
Input Question
      ↓
Tokenization
      ↓
Word-to-Index Conversion
      ↓
Embedding Layer
      ↓
Simple RNN
      ↓
Final Hidden State
      ↓
Fully Connected Layer
      ↓
Predicted Answer Token
```

## ⚙️ Model Components

The model consists of:

* **Embedding Layer:** Converts word indices into dense vector representations.
* **Simple RNN:** Processes the sequence of word embeddings.
* **Fully Connected Layer:** Maps the final hidden state to the vocabulary size.

```python id="w84nq2"
Embedding(vocab_size, 50)
        ↓
RNN(50, 64)
        ↓
Linear(64, vocab_size)
```

## 🛠️ Technologies Used

* Python
* PyTorch
* Pandas
* NumPy

## 📂 Dataset

The project uses a custom CSV dataset containing:

* `question`
* `answer`

The vocabulary is built by combining tokens from both the questions and answers.

## 🔄 Text Preprocessing

The preprocessing pipeline:

1. Converts text to lowercase.
2. Removes selected punctuation.
3. Splits text into individual tokens.
4. Builds a vocabulary from questions and answers.
5. Converts tokens into numerical indices.
6. Uses `<UNK>` for unknown words.

## 🏋️ Training

The model is trained using:

* **Optimizer:** Adam
* **Learning Rate:** `0.001`
* **Loss Function:** CrossEntropyLoss
* **Epochs:** `20`

## 🔮 Prediction

For a new question:

1. The question is tokenized.
2. Tokens are converted into vocabulary indices.
3. The model generates output logits.
4. Softmax converts logits into probabilities.
5. The token with the highest probability is selected.
6. If confidence is below the defined threshold, the model can return `"I don't know"`.

## 📁 Project Structure

```text id="y3h6mf"
Simple-RNN-QA-System/
│
├── simple_rnn_qa.ipynb
├── 100_Unique_QA_Dataset.csv
└── README.md
```

## 🎯 Key Concepts

* Natural Language Processing
* Text Tokenization
* Vocabulary Building
* Word Embeddings
* Custom PyTorch Dataset
* DataLoader
* Recurrent Neural Networks
* CrossEntropy Loss
* Text Classification Concepts
* Model Training and Inference

## 👨‍💻 Author

**Shayan Ahmed**

AI / Machine Learning Engineer

📧 [iamshayanjaved@gmail.com](mailto:iamshayanjaved@gmail.com)
