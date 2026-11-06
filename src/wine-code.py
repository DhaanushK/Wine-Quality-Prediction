import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    concatenate,
)
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Train the hybrid wine quality model.")
parser.add_argument(
    "--wine",
    choices=("red", "white"),
    default="white",
    help="Choose the wine dataset to train and evaluate.",
)
args = parser.parse_args()

seed = 8

project_root = Path(__file__).resolve().parents[1]
dataset_path = project_root / "dataset" / f"winequality-{args.wine}-balanced.csv"
data = pd.read_csv(dataset_path)

y = data["quality"].round().astype(int)
X = data.drop(columns="quality")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.to_numpy().reshape(-1, 1)).toarray()

X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)


def create_cnn_branch(input_layer):
    x = Conv1D(
        32, kernel_size=2, activation="relu", padding="same", kernel_regularizer=l2(0.01)
    )(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(
        64, kernel_size=2, activation="relu", padding="same", kernel_regularizer=l2(0.01)
    )(x)
    x = MaxPooling1D(pool_size=2)(x)
    return Flatten()(x)


split = X_train_cnn.shape[1] // 2
input1 = Input(shape=(split, 1))
input2 = Input(shape=(X_train_cnn.shape[1] - split, 1))

branch1 = create_cnn_branch(input1)
branch2 = create_cnn_branch(input2)
merged = concatenate([branch1, branch2])

x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(merged)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
cnn_output = Dense(y_train_encoded.shape[1], activation="softmax")(x)

cnn_model = Model(inputs=[input1, input2], outputs=cnn_output)
cnn_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
cnn_early_stop = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
cnn_model.fit(
    [X_train_cnn[:, :split, :], X_train_cnn[:, split:, :]],
    y_train_encoded,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[cnn_early_stop],
    verbose=1,
)

cnn_inputs_test = [X_test_cnn[:, :split, :], X_test_cnn[:, split:, :]]
cnn_test_accuracy = cnn_model.evaluate(cnn_inputs_test, y_test_encoded, verbose=0)[1]

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64), max_iter=1000, random_state=seed
)
mlp.fit(X_train_scaled, y_train)
mlp_train_probabilities = mlp.predict_proba(X_train_scaled)
mlp_test_probabilities = mlp.predict_proba(X_test_scaled)
mlp_test_accuracy = mlp.score(X_test_scaled, y_test)

mlp_input = Input(shape=(mlp_train_probabilities.shape[1],), name="mlp_probabilities")
cnn_probabilities = Flatten()(cnn_output)
combined = concatenate([cnn_probabilities, mlp_input])
combined = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(combined)
combined = Dropout(0.5)(combined)
combined = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(combined)
combined = Dropout(0.5)(combined)
combined = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(combined)
combined = Dropout(0.5)(combined)
final_output = Dense(y_train_encoded.shape[1], activation="softmax")(combined)

combined_model = Model(
    inputs=[input1, input2, mlp_input],
    outputs=final_output,
)
combined_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
combined_early_stop = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
combined_model.fit(
    [
        X_train_cnn[:, :split, :],
        X_train_cnn[:, split:, :],
        mlp_train_probabilities,
    ],
    y_train_encoded,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[combined_early_stop],
    verbose=1,
)

test_probabilities = combined_model.predict(
    [cnn_inputs_test[0], cnn_inputs_test[1], mlp_test_probabilities], verbose=0
)
y_pred = np.argmax(test_probabilities, axis=1)
y_true = np.argmax(y_test_encoded, axis=1)
combined_test_accuracy = accuracy_score(y_true, y_pred)

print(f"CNN test accuracy ({args.wine} wine): {cnn_test_accuracy:.4%}")
print(f"MLP test accuracy ({args.wine} wine): {mlp_test_accuracy:.4%}")
print(f"Combined test accuracy ({args.wine} wine): {combined_test_accuracy:.4%}")

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred)).plot()
plt.title(f"Combined Model Confusion Matrix ({args.wine.title()} Wine)")
plt.tight_layout()
plt.show()
