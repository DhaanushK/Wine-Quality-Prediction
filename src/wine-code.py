import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
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

# The balanced CSVs store pre-scaled inputs. Keep the matching transforms for
# accepting raw UCI-style measurements in the interactive app.
RAW_INPUT_TRANSFORMS = {
    "red": {
        "method": "minmax",
        "min": {
            "fixed acidity": 4.6,
            "volatile acidity": 0.12,
            "citric acid": 0.0,
            "residual sugar": 0.9,
            "chlorides": 0.012,
            "free sulfur dioxide": 1.0,
            "total sulfur dioxide": 6.0,
            "density": 0.99007,
            "pH": 2.74,
            "sulphates": 0.33,
            "alcohol": 8.4,
        },
        "max": {
            "fixed acidity": 15.9,
            "volatile acidity": 1.58,
            "citric acid": 1.0,
            "residual sugar": 15.5,
            "chlorides": 0.611,
            "free sulfur dioxide": 72.0,
            "total sulfur dioxide": 289.0,
            "density": 1.00369,
            "pH": 4.01,
            "sulphates": 2.0,
            "alcohol": 14.9,
        },
    },
    "white": {
        "method": "standard",
        "mean": {
            "fixed acidity": 6.854787668436097,
            "volatile acidity": 0.27824111882400976,
            "citric acid": 0.33419150673744386,
            "residual sugar": 6.391414863209474,
            "chlorides": 0.04577235606369946,
            "free sulfur dioxide": 35.30808493262556,
            "total sulfur dioxide": 138.36065741118824,
            "density": 0.9940273764801959,
            "pH": 3.1882666394446715,
            "sulphates": 0.48984687627603113,
            "alcohol": 10.514267047774602,
        },
        "scale": {
            "fixed acidity": 0.8437820791264506,
            "volatile acidity": 0.10078425854188974,
            "citric acid": 0.12100744957029214,
            "residual sugar": 5.071539989333933,
            "chlorides": 0.02184573768505638,
            "free sulfur dioxide": 17.005401105808414,
            "total sulfur dioxide": 42.49372602475034,
            "density": 0.0029906015821480275,
            "pH": 0.15098518431212068,
            "sulphates": 0.1141141831056649,
            "alcohol": 1.2304949365418656,
        },
    },
}

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

artifacts_dir = project_root / "artifacts"
artifacts_dir.mkdir(exist_ok=True)
pipeline_path = artifacts_dir / f"{args.wine}-pipeline.joblib"
model_path = artifacts_dir / f"{args.wine}-combined.keras"
joblib.dump(
    {
        "scaler": scaler,
        "mlp": mlp,
        "classes": encoder.categories_[0].astype(int),
        "features": list(X.columns),
        "split": split,
        "raw_input_transform": RAW_INPUT_TRANSFORMS[args.wine],
    },
    pipeline_path,
)
combined_model.save(model_path)

print(f"CNN test accuracy ({args.wine} wine): {cnn_test_accuracy:.4%}")
print(f"MLP test accuracy ({args.wine} wine): {mlp_test_accuracy:.4%}")
print(f"Combined test accuracy ({args.wine} wine): {combined_test_accuracy:.4%}")
print(f"Saved inference model to: {model_path}")

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred)).plot()
plt.title(f"Combined Model Confusion Matrix ({args.wine.title()} Wine)")
plt.tight_layout()
plt.show()
