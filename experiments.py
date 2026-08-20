# Project Description: Next Word Prediction Using LSTM
# Project Overview:
"""
This project aims to develop a deep learning model for predicting the next word in a given sequence of words. 
The model is built using Long Short-Term Memory (LSTM) networks, which are well-suited for sequence prediction tasks.
 The project includes the following steps:

1- Data Collection: We use the text of Shakespeare's "Hamlet" as our dataset. 
This rich, complex text provides a good challenge for our model.

2- Data Preprocessing: The text data is tokenized, converted into sequences, and padded to ensure uniform input lengths. 
The sequences are then split into training and testing sets.

3- Model Building: An LSTM model is constructed with an embedding layer, two LSTM layers, and a dense output layer with a softmax activation function to predict the probability of the next word.

4- Model Training: The model is trained using the prepared sequences, with early stopping implemented to prevent overfitting. 
Early stopping monitors the validation loss and stops training when the loss stops improving.

5- Model Evaluation: The model is evaluated using a set of example sentences to test its ability to predict the next word accurately.

6- Deployment: A Streamlit web application is developed to allow users to input a sequence of words and get the predicted next word in real-time.

"""

# Data Collection 
# import nltk
# nltk.download('gutenberg')
# from nltk.corpus import gutenberg

# # load the dataset
# data=gutenberg.raw('shakespeare-hamlet.txt')

# # save to a file
# with open('hamlet.txt', 'w') as file:
#     file.write(data)


# Data Preprocessing

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#load the Dataset

with open('hamlet.txt', 'r')as file:
    text=file.read().lower()

# Tokenize the text-creating indexes for words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words=len(tokenizer.word_index)+1
# print(total_words)
# print(tokenizer.word_index)

# Create input sequences
input_sequences =[]
for line in text.split('\n'):
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
            n_gram_sequence=token_list[:i+1]
            input_sequences.append(n_gram_sequence)

# print(input_sequences)

# Pad Sequences 
max_sequence_len = max([len(x) for x in input_sequences])

# print(max_sequence_len)

input_sequences=np.array(pad_sequences(input_sequences,max_sequence_len,padding='pre'))

# print(input_sequences[0])
# print(input_sequences[1])

##create predicitors and label
import tensorflow as tf
x,y=input_sequences[:,:-1],input_sequences[:,-1]
# print(y)
y=tf.keras.utils.to_categorical(y,num_classes=total_words)
# print(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# Train our LSTM RNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Embedding,LSTM,Dense,Dropout,GRU

# Define the model
model=Sequential()
model.add(Input(shape=(max_sequence_len - 1,)))  # ✅ clean way

#input_length is deprecated use input_shape like this in a tuple
# model.add(Embedding(total_words,100,input_shape=(max_sequence_len-1,))) 

model.add(Embedding(total_words,100))
model.add(LSTM(150,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation="softmax"))

# #Compile the model
model.compile(
     loss="categorical_crossentropy",
     optimizer='adam',
     metrics=['accuracy'])

# print("LSTM",model.summary())

## Train the model
history=model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),verbose=1,callbacks=[early_stopping])


## Save the model
model.save("next_word_lstm.h5")
## Save the tokenizer
import pickle
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)



# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None



# Usage to test
# input_text="To be or not to be"
# print(f"Input text:{input_text}")
# max_sequence_len=model.input_shape[1]+1
# next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
# print(f"Next Word PRediction:{next_word}")





