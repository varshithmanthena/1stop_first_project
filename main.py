import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Dropout
import numpy as np
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Download necessary NLTK data for nlp tasks
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Preprocess the text by lowercasing, removing non-alphabetic characters, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load dataset
data = pd.read_csv(r'C:\Users\hvb23\Downloads\1stop_first_project\dataset_twitter\twitter_training.csv')
data['Text'] = data['Text'].astype(str).apply(clean_text)

# Encode Sentiments
label_map = {'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3}
data['Sentiment'] = data['Sentiment'].map(label_map)

# Tokenization
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Text'])

# Save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(data['Text'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['Sentiment'].values, test_size=0.2, random_state=42)


model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=100, mask_zero=True),  # (batch_size, 100, 128)
    Conv1D(64, kernel_size=5, activation='relu'),  # (batch_size, 96, 64)
    MaxPooling1D(pool_size=2),  #  Keeps sequence length (batch_size, 48, 64)
    Bidirectional(LSTM(64, return_sequences=True)),  # (batch_size, 48, 64)
    Bidirectional(LSTM(32)),  # (batch_size, 32)
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_text_classification_model.keras', save_best_only=True)

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping, model_checkpoint])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Save final model lets save  this with .keras.
model.save('final_text_classification_model.keras')
