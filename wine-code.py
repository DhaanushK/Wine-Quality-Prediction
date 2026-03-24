import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('/content/winequality-white-balanced (1).csv')

y = data['quality']
X = data.drop('quality', axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1)).toarray()

# CNN reshape
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

def create_cnn_branch(input_layer):
    x = Conv1D(32, 2, activation='relu', padding='same', kernel_regularizer=l2(0.01))(input_layer)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = MaxPooling1D(2)(x)
    return Flatten()(x)

split = X_train_cnn.shape[1] // 2
input1 = Input(shape=(split, 1))
input2 = Input(shape=(X_train_cnn.shape[1]-split, 1))

branch1 = create_cnn_branch(input1)
branch2 = create_cnn_branch(input2)

merged = concatenate([branch1, branch2])

x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(merged)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)

output = Dense(y_train_encoded.shape[1], activation='softmax')(x)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    [X_train_cnn[:, :split, :], X_train_cnn[:, split:, :]],
    y_train_encoded,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Plot accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Plot loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# Predictions
y_pred_probs = model.predict([X_test_cnn[:, :split, :], X_test_cnn[:, split:, :]])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_encoded, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
