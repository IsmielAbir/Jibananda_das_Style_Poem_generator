import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import random

text = open('Dataset.txt').read()[:500000]
chars = list(set(text))
data_size, vocab_size = len(text), len(chars)
char_indices = { ch:i for i,ch in enumerate(chars) }
indices_char = { i:ch for i,ch in enumerate(chars) }

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)))
y = np.zeros((len(sentences), len(chars)))
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = tf.keras.Sequential(
    [
        layers.InputLayer(input_shape=(maxlen, len(chars))),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.Dense(len(chars), activation="softmax")
    ]
)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

model.summary()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

try:
    epochs = 40
    batch_size = 128

    data_size = len(x)
    subset_size = data_size  # Use 1/4 of the data

    for epoch in range(epochs):
        # Randomly select a subset of indices for this epoch
        indices = random.sample(range(data_size), subset_size)
        x_subset = x[indices]
        y_subset = y[indices]

        model.fit(x_subset, y_subset, batch_size=batch_size, epochs=1)
        print()
        print("==> Generating text after epoch: %d" % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0]:
            print("==> Diversity:", diversity)

            generated = ""
            sentence = text[start_index: start_index + maxlen]
            print('==> Generating with seed:\n' + sentence)
            in_sentence = sentence

            for i in range(300):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.0
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                sentence = sentence[1:] + next_char
                generated += next_char

            print("==> Generated: \n", in_sentence + generated)
            print()

except Exception as e:
    print("Error:", e)

generated = ""
diversity = 0.5
sentence = "হাজার বছর ধরে আমি "
if len(sentence) > maxlen:
  sentence = sentence[:maxlen]
else:
  pad = " " * (maxlen - len(sentence))
  sentence = pad + sentence
print('==> Generating with seed:\n' + sentence)
in_sentence = sentence

for i in range(400):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.0
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    sentence = sentence[1:] + next_char
    generated += next_char

print("==> Generated: \n", in_sentence + generated)
print()