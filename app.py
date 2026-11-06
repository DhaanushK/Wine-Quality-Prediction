import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import numpy as np
from keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
QUALITY_CUTOFF = 6

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

COLORS = {
    "background": "#f4f1eb",
    "card": "#ffffff",
    "ink": "#1c302d",
    "muted": "#61716d",
    "accent": "#9e4147",
    "accent_dark": "#783238",
    "line": "#dce2dc",
    "green": "#315b4b",
}


class WineQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wine Quality Predictor")
        self.root.configure(bg=COLORS["background"])
        self.root.geometry("820x800")
        self.root.minsize(720, 740)

        self.wine_type = tk.StringVar(value="white")
        self.entries = {}
        self.pipeline_cache = {}
        self.model_cache = {}
        self.quality_result = tk.StringVar(value="Enter a sample to see its prediction")
        self.tier_result = tk.StringVar(value="")
        self.probability_result = tk.StringVar(value="")

        self._configure_styles()
        self._build_layout()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Wine.TCombobox",
            fieldbackground=COLORS["card"],
            background=COLORS["card"],
            foreground=COLORS["ink"],
            padding=7,
        )
        style.configure(
            "Wine.TEntry",
            fieldbackground=COLORS["card"],
            foreground=COLORS["ink"],
            padding=8,
        )

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg=COLORS["ink"], padx=28, pady=14)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            header,
            text="WINE QUALITY",
            bg=COLORS["ink"],
            fg="#e7c4a2",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="A sample-based prediction",
            bg=COLORS["ink"],
            fg="#ffffff",
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w", pady=(5, 2))
        tk.Label(
            header,
            text="Choose a wine and enter its laboratory measurements.",
            bg=COLORS["ink"],
            fg="#d0d9d4",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        content = tk.Frame(self.root, bg=COLORS["background"], padx=24, pady=10)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)

        selector = tk.Frame(content, bg=COLORS["background"])
        selector.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(
            selector,
            text="Wine type",
            bg=COLORS["background"],
            fg=COLORS["ink"],
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(0, 12))
        self.wine_selector = ttk.Combobox(
            selector,
            textvariable=self.wine_type,
            values=("red", "white"),
            state="readonly",
            width=14,
            style="Wine.TCombobox",
        )
        self.wine_selector.pack(side="left")

        form = tk.Frame(
            content,
            bg=COLORS["card"],
            highlightbackground=COLORS["line"],
            highlightthickness=1,
            padx=20,
            pady=8,
        )
        form.grid(row=1, column=0, sticky="ew")
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, weight=1)
        tk.Label(
            form,
            text="Sample measurements",
            bg=COLORS["card"],
            fg=COLORS["ink"],
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 3))
        tk.Label(
            form,
            text="Enter the 11 raw measurements; scaling and derived features are handled automatically.",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        for index, feature in enumerate(FEATURES):
            column = index % 3
            row = 2 + index // 3
            field = tk.Frame(form, bg=COLORS["card"])
            field.grid(
                row=row,
                column=column,
                sticky="ew",
                padx=(0, 8) if column == 0 else ((8, 4) if column == 1 else (4, 0)),
                pady=3,
            )
            field.columnconfigure(0, weight=1)
            label = "pH" if feature == "pH" else feature.title()
            tk.Label(
                field,
                text=label,
                bg=COLORS["card"],
                fg=COLORS["muted"],
                font=("Segoe UI", 9),
            ).grid(row=0, column=0, sticky="w", pady=(0, 3))
            entry = ttk.Entry(field, style="Wine.TEntry")
            entry.grid(row=1, column=0, sticky="ew")
            self.entries[feature] = entry

        actions = tk.Frame(content, bg=COLORS["background"])
        actions.grid(row=2, column=0, sticky="ew", pady=10)
        tk.Button(
            actions,
            text="Predict quality",
            command=self.predict,
            bg=COLORS["accent"],
            fg="white",
            activebackground=COLORS["accent_dark"],
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        ).pack(side="left")
        tk.Button(
            actions,
            text="Clear values",
            command=self.clear,
            bg=COLORS["background"],
            fg=COLORS["ink"],
            activebackground=COLORS["line"],
            relief="flat",
            padx=16,
            pady=10,
            font=("Segoe UI", 10),
            cursor="hand2",
        ).pack(side="left", padx=8)

        result = tk.Frame(
            content,
            bg=COLORS["card"],
            highlightbackground=COLORS["line"],
            highlightthickness=1,
            padx=20,
            pady=8,
        )
        result.grid(row=3, column=0, sticky="ew")
        tk.Label(
            result,
            text="PREDICTION",
            bg=COLORS["card"],
            fg=COLORS["accent"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        tk.Label(
            result,
            textvariable=self.quality_result,
            bg=COLORS["card"],
            fg=COLORS["ink"],
            font=("Segoe UI", 16, "bold"),
            wraplength=640,
            justify="left",
        ).pack(anchor="w", pady=(3, 2))
        tk.Label(
            result,
            textvariable=self.tier_result,
            bg=COLORS["card"],
            fg=COLORS["green"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            result,
            textvariable=self.probability_result,
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 0))

        note = (
            "Quality estimate only. The model does not test for contamination, spoilage, "
            "or whether a wine is safe to drink."
        )
        tk.Label(
            content,
            text=note,
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
            wraplength=690,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(10, 0))
        tk.Label(
            content,
            text=f"Project tier rule: predicted quality ≥ {QUALITY_CUTOFF} is higher; below {QUALITY_CUTOFF} is lower.",
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).grid(row=5, column=0, sticky="w", pady=(4, 0))

    def _load_models(self, wine_type):
        if wine_type not in self.pipeline_cache:
            pipeline_path = ARTIFACTS_DIR / f"{wine_type}-pipeline.joblib"
            model_path = ARTIFACTS_DIR / f"{wine_type}-combined.keras"
            if not pipeline_path.exists() or not model_path.exists():
                raise FileNotFoundError(
                    f"The {wine_type} model has not been trained yet. "
                    f"Run: python .\\src\\wine-code.py --wine {wine_type}"
                )
            self.pipeline_cache[wine_type] = joblib.load(pipeline_path)
            self.model_cache[wine_type] = load_model(model_path, compile=False)
        return self.pipeline_cache[wine_type], self.model_cache[wine_type]

    def predict(self):
        try:
            wine_type = self.wine_type.get()
            pipeline, model = self._load_models(wine_type)
            feature_names = pipeline["features"]
            values = []
            for name in FEATURES:
                raw = self.entries[name].get().strip()
                if not raw:
                    raise ValueError(f"Enter a value for {name}.")
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number.")
                values.append(value)

            raw_values = dict(zip(FEATURES, values))
            transform = pipeline["raw_input_transform"]
            if transform["method"] == "minmax":
                scaled_features = {
                    name: (value - transform["min"][name])
                    / (transform["max"][name] - transform["min"][name])
                    for name, value in raw_values.items()
                }
            elif transform["method"] == "standard":
                scaled_features = {
                    name: (value - transform["mean"][name]) / transform["scale"][name]
                    for name, value in raw_values.items()
                }
            else:
                raise ValueError("Unknown raw-input preprocessing configuration.")

            derived_features = {
                "alcohol_sulphate_interaction": (
                    scaled_features["alcohol"] * scaled_features["sulphates"]
                )
            }
            model_values = []
            for name in feature_names:
                if name in scaled_features:
                    model_values.append(scaled_features[name])
                elif name in derived_features:
                    model_values.append(derived_features[name])
                else:
                    raise ValueError(f"The model input feature '{name}' is not supported.")

            sample = np.asarray(model_values, dtype=np.float32).reshape(1, -1)
            scaled = pipeline["scaler"].transform(sample)
            mlp_probabilities = pipeline["mlp"].predict_proba(scaled)
            split = pipeline["split"]
            cnn_values = scaled.reshape(1, scaled.shape[1], 1)
            probabilities = model.predict(
                [
                    cnn_values[:, :split, :],
                    cnn_values[:, split:, :],
                    mlp_probabilities,
                ],
                verbose=0,
            )[0]

            classes = pipeline["classes"]
            predicted_index = int(np.argmax(probabilities))
            predicted_quality = int(classes[predicted_index])
            top_probability = float(probabilities[predicted_index])
            tier = "Higher predicted quality" if predicted_quality >= QUALITY_CUTOFF else "Lower predicted quality"

            self.quality_result.set(f"Predicted quality score: {predicted_quality}")
            self.tier_result.set(tier)
            self.probability_result.set(f"Top-class model probability: {top_probability:.1%}")
        except FileNotFoundError as error:
            messagebox.showerror("Model not trained", str(error), parent=self.root)
        except ValueError as error:
            messagebox.showerror("Check the sample values", str(error), parent=self.root)
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error), parent=self.root)

    def clear(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.quality_result.set("Enter a sample to see its prediction")
        self.tier_result.set("")
        self.probability_result.set("")


if __name__ == "__main__":
    window = tk.Tk()
    WineQualityApp(window)
    window.mainloop()
