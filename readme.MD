# 🎬 Shakespeare Next Word Prediction using LSTM & GRU

A Deep Learning and Natural Language Processing project that predicts the **next word in a text sequence** using recurrent neural networks trained on William Shakespeare's *Hamlet*.

This project now includes **both LSTM and GRU models**, allowing sequence prediction using two popular RNN architectures.

---

## 📌 Project Name

### **Shakespeare Next Word Predictor — LSTM & GRU**

Recommended GitHub repository name:

```text
shakespeare-next-word-prediction-lstm-gru
```

---

## 📖 Project Overview

The objective of this project is to build a **Next Word Prediction System** using Deep Learning.

The model learns word patterns from Shakespeare's *Hamlet* and predicts the most likely next word when a user enters a sequence of words.

### Example

```text
Input:
To be or not to

Prediction:
be
```

The project demonstrates an end-to-end NLP workflow:

```text
Hamlet Text
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
N-Gram Sequence Generation
     ↓
Padding
     ↓
Train/Test Split
     ↓
LSTM / GRU Model
     ↓
Model Training
     ↓
Save Model + Tokenizer
     ↓
Streamlit Application
     ↓
Next Word Prediction
```

---

## 📚 Dataset

The project uses **William Shakespeare's Hamlet** as the text corpus.

The text is stored locally in:

```text
hamlet.txt
```

The corpus is tokenized so that individual words can be converted into numerical IDs that can be processed by the neural network.

---

# 🔄 Data Preprocessing

## 1. Tokenization

The text is converted into sequences of integer word IDs using a tokenizer.

For example:

```text
To be or not to be
```

is converted into numerical values representing the words.

The tokenizer is later saved as:

```text
tokenizer.pickle
```

This is important because the Streamlit application needs to convert new user input into the same numerical representation used during training.

---

## 2. Creating N-Gram Sequences

The project creates progressively longer sequences from the text.

For example:

```text
To
To be
To be or
To be or not
To be or not to
To be or not to be
```

The final word in each sequence becomes the target that the model has to predict.

---

## 3. Padding

Since the generated sequences have different lengths, padding is used to make them the same length.

The model input uses:

```python
Input(shape=(max_sequence_len - 1,))
```

The `-1` is used because the final word of every sequence is separated as the target/output word.

---

# 🤖 Model Architecture

This project contains **two RNN architectures**:

1. LSTM
2. GRU

Both models use an Embedding layer followed by recurrent layers and a final Softmax layer for next-word prediction.

---

# 🧠 LSTM Model

The LSTM architecture is:

```text
Input
  ↓
Embedding
  ↓
LSTM (150 units)
  ↓
Dropout (20%)
  ↓
LSTM (100 units)
  ↓
Dense + Softmax
  ↓
Predicted Next Word
```

### LSTM Code

```python
model = Sequential()

model.add(Input(shape=(max_sequence_len - 1,)))

model.add(
    Embedding(
        input_dim=total_words,
        output_dim=100
    )
)

model.add(
    LSTM(
        150,
        return_sequences=True
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(100)
)

model.add(
    Dense(
        total_words,
        activation="softmax"
    )
)
```

---

# 🔵 GRU Model

The project also includes a **GRU (Gated Recurrent Unit)** model.

The GRU architecture is:

```text
Input
  ↓
Embedding
  ↓
GRU (150 units)
  ↓
Dropout (20%)
  ↓
GRU (100 units)
  ↓
Dense + Softmax
  ↓
Predicted Next Word
```

### GRU Code

```python
model = Sequential()

model.add(
    Input(
        shape=(max_sequence_len - 1,)
    )
)

model.add(
    Embedding(
        input_dim=total_words,
        output_dim=100
    )
)

model.add(
    GRU(
        150,
        return_sequences=True
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    GRU(100)
)

model.add(
    Dense(
        total_words,
        activation="softmax"
    )
)
```

---

# ⚙️ Model Compilation

Both the LSTM and GRU models are compiled using:

```python
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

### Loss Function

```text
categorical_crossentropy
```

is used for multi-class classification because the model predicts one word from the complete vocabulary.

### Optimizer

```text
Adam
```

is used to update the model weights during training.

### Output Activation

```text
Softmax
```

is used in the final Dense layer.

The Softmax layer produces a probability for each possible word in the vocabulary.

---

# 🔍 LSTM vs GRU

The project allows experimentation with both architectures.

| Feature | LSTM | GRU |
|---|---|---|
| Architecture | More complex | Simpler |
| Memory mechanism | Cell state + hidden state | Hidden state |
| Gates | Input, Forget, Output | Update, Reset |
| Parameters | Generally more | Generally fewer |
| Training | Can be slower | Often faster |
| Sequence learning | Excellent | Excellent |

The actual performance should be evaluated using the training/validation results from the experiments rather than assuming one model is always better.

---

# 🔮 Next Word Prediction

After training, the model predicts the next word by:

1. Receiving the user's text.
2. Converting the text into token IDs.
3. Preparing the sequence with the required length.
4. Passing the sequence to the trained model.
5. Obtaining probabilities for all words in the vocabulary.
6. Selecting the word with the highest predicted probability.

The prediction uses:

```python
predicted_word_index = np.argmax(predicted, axis=1)
```

The predicted index is then converted back to the corresponding word using the tokenizer.

---

# 💾 Saved Model and Tokenizer

The trained model is saved as:

```text
next_word_lstm.h5
```

The tokenizer is saved as:

```text
tokenizer.pickle
```

If you save the GRU model separately, a name such as:

```text
next_word_gru.h5
```

can be used.

> Keep the model file and tokenizer together with the application when running the prediction system.

---

# 🌐 Streamlit Application

The project includes a Streamlit interface where the user can enter a sequence of words and request the next-word prediction.

The application loads the trained model and tokenizer:

```python
model = load_model('next_word_lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
```

The user enters text and clicks:

```text
Predict Next Word
```

The application then displays the predicted word.

---

# 📁 Project Structure

```text
shakespeare-next-word-prediction-lstm-gru/
│
├── app(1).py
├── experiment.ipynb
├── hamlet.txt
├── next_word_lstm.h5
├── tokenizer.pickle
├── requirements.txt
└── README.md
```

If you also save the GRU model separately:

```text
shakespeare-next-word-prediction-lstm-gru/
│
├── app(1).py
├── experiment.ipynb
├── hamlet.txt
├── next_word_lstm.h5
├── next_word_gru.h5
├── tokenizer.pickle
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep Learning |
| LSTM | Sequence modelling |
| GRU | Sequence modelling |
| NLP | Text processing |
| Tokenization | Convert words to numerical IDs |
| NumPy | Numerical operations |
| Scikit-learn | Dataset splitting |
| NLTK | Text corpus processing |
| Streamlit | Web application |
| Pickle | Tokenizer serialization |

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Go to the project directory:

```bash
cd shakespeare-next-word-prediction-lstm-gru
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If you are creating the environment manually, you can install the main packages with:

```bash
pip install tensorflow numpy nltk scikit-learn streamlit
```

---

# ▶️ Run the Notebook

Open:

```text
experiment.ipynb
```

Run the notebook cells in order to:

- Load the Hamlet corpus
- Preprocess the text
- Tokenize the text
- Generate N-gram sequences
- Pad the sequences
- Prepare features and targets
- Train the LSTM model
- Train the GRU model
- Evaluate the models
- Save the trained model
- Save the tokenizer

---

# 🚀 Run the Streamlit Application

Make sure the required model and tokenizer files are in the project directory.

For the LSTM version:

```bash
streamlit run app(1).py
```

The application will open in your browser.

---

# 💡 Example Inputs

Try inputs such as:

```text
To be or not to
```

```text
What a piece of
```

```text
The king is
```

```text
There is nothing
```

The model predicts the next word based on patterns learned from the training corpus.

---

# 🎯 Key Learning Outcomes

This project demonstrates practical knowledge of:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Vocabulary creation
- N-gram sequence generation
- Sequence padding
- Feature and target creation
- Word embeddings
- Recurrent Neural Networks
- LSTM
- GRU
- Dropout
- Softmax
- Categorical cross-entropy
- Adam optimizer
- Model training
- Model evaluation
- Model serialization
- Streamlit application development
- Next-word prediction

---

# 🔬 Future Improvements

Possible improvements include:

- Compare LSTM and GRU accuracy using the same dataset and training configuration.
- Display LSTM vs GRU training/validation graphs.
- Display Top-5 next-word predictions.
- Add probability scores for predictions.
- Use temperature-based text generation.
- Train using a larger Shakespeare corpus.
- Experiment with Bidirectional LSTM/GRU.
- Experiment with Transformer-based language models.
- Improve the Streamlit UI.
- Deploy the application online.

---

# ⭐ Project Highlights

```text
NLP
+
Deep Learning
+
RNN
+
LSTM
+
GRU
+
Word Embeddings
+
Streamlit
```

This project demonstrates an end-to-end **Natural Language Processing and Deep Learning workflow**, from raw Shakespeare text to an interactive next-word prediction application.

---

# 👨‍💻 Author

**Saif Ali Khan**

**Bachelor of Engineering — Artificial Intelligence & Machine Learning**

GitHub:

```text
https://github.com/SaifAK45
```

---
