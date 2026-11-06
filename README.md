# Wine Quality Prediction using Deep Learning

## Overview
This project predicts wine quality using a hybrid deep learning model (MC-CNN + MLP).

## Tech Stack
- Python
- TensorFlow
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## Workflow
1. Data preprocessing (StandardScaler, normalization)
2. Train-test split (80:20)
3. Hybrid multichannel CNN and MLP training with dropout (0.5)
4. Evaluation and visualization

## Results
- The paper reports 87.6% accuracy for red wine and 90% for white wine.
- The historical project screenshot records 87.65% combined test accuracy for red wine.
- Latest local run of the CNN + MLP pipeline: combined test accuracy is 87.41% for red wine and 90.58% for white wine. The standalone MLP scores are 87.65% and 90.58%, respectively. The CNN has unseeded random initialization, so results can vary slightly between runs.
- The PDF says to remove duplicates, but the original notebook and saved red result use the balanced CSV as provided. The historical pipeline is retained here to compare against the saved result.

## Key Features
- Feature scaling
- Overfitting control
- Visualization of correlations

## Dataset
The project includes balanced red and white wine CSV files in `dataset/`. Choose which dataset to train with `--wine red` or `--wine white`.

## Run locally (Windows PowerShell)

From the project folder, create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then run either dataset separately:

```powershell
python .\src\wine-code.py --wine red
python .\src\wine-code.py --wine white
```

The training script reads the selected bundled balanced dataset using a path relative to the script, rounds the quality target, and prints CNN, MLP, and combined test accuracies. TensorFlow runs on CPU on native Windows. The CNN's random initialization is unseeded, as in the historical notebook, so its combined score may vary slightly between runs.

## Interactive prediction app

Train both models once to create local inference artifacts:

```powershell
python .\src\wine-code.py --wine red
python .\src\wine-code.py --wine white
```

Then start the desktop form:

```powershell
python .\app.py
```

Choose red or white wine, replace the example in each of the 11 measurement fields with the sample's lab reading, and select **Predict quality**. Each field shows its measurement unit and an example value. The form applies the matching red or white scaling and computes the engineered alcohol-sulphates interaction automatically. It displays the predicted quality score and a project-defined tier (scores of 6 or higher are labeled higher quality). This is a quality estimate only; it does not determine food safety or edibility. Trained model files are stored locally under `artifacts/` and are excluded from Git.
