import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


# Example with a CSV file (replace with your dataset)
#df = pd.read_csv('text_data.csv')  # Assuming columns: 'text' and 'label'

# For demonstration, let's create a simple dataset if you don't have one
data = {'text': ['I love this product!', 'This is terrible.', 'Great experience.', 'Worst service ever.', 'Highly recommended.'],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive']}
df = pd.DataFrame(data)

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

# Tokenize text
tokenizer = Tokenizer(num_words=5000)  # Consider top 5000 words
tokenizer.fit_on_texts(df['processed_text'])
sequences = tokenizer.texts_to_sequences(df['processed_text'])
print(sequences)

# Pad sequences to ensure uniform length
max_len = 10  # Maximum sequence length
X = pad_sequences(sequences, maxlen=max_len)
y = df['label_encoded'].values

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)) ##output shape=(batch_size,128-embedding,100-input_vector_length)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) ##batch_size,128
model.add(Dense(64, activation='relu'))  ##batch_size,64
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # For binary classification batch_size,1

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))  ##output= (batch_size,128-embedd,100-input_seq)
model.add(Flatten())  # Destroys sequence! # converts 3d tensor to 2d tensor for dense (batch_size, 128*100)
model.add(Dense(128, activation='relu'))  # Treats words independently (128 neurons -batch_size,128*100)
model.add(Dense(1, activation='sigmoid')) #binary output

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return 'positive' if prediction > 0.5 else 'negative'

# Example prediction
print(predict_sentiment("This movie was fantastic!"))  # Should output 'positive'

