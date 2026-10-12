# Audio_Grammar_Scoring

An audio-based machine learning project that predicts spoken English grammar scores from speech recordings. It explores transcript-based language models, acoustic representations, and ensemble learning for continuous score prediction.

## Project Overview

The goal is to predict a grammar score for each spoken-English recording.

The repository contains two notebooks:

| Notebook | Description |
|---|---|
| `Grammar_Scoring_Experiments.ipynb` | Full experimentation and model-development workflow, including exploration, model comparisons, and ensemble experiments. |
| `Grammar_Scoring_Engine.ipynb` | Clean evaluation and inference workflow with visualizations, error analysis, ensemble evaluation, and submission generation. |

The clean notebook reuses saved model predictions and embeddings where available, avoiding unnecessary reruns of expensive neural-model training and feature extraction.

## Model Architecture

```text
                         Input speech audio
                                  |
                +-----------------+-----------------+
                |                 |                 |
        Whisper canonical  Whisper verbatim   Wav2Vec2 CTC
          transcription     transcription    transcription
                |                 |                 |
                +-----------------+-----------------+
                                  |
                         DeBERTa regressors
                                  |
                       Text-based predictions
                                  |
                                  +----------------------+
                                                         |
Audio representations                                    |
        |                                                |
   +----+--------------------+                           |
   |                         |                           |
WavLM embeddings       Whisper embeddings                |
   |                         |                           |
 RBF-SVR                   RBF-SVR                      |
   |                         |                           |
   +-------------------------+---------------------------+
                                  |
                      Non-negative linear stack
                                  |
                       Final grammar score
                                  |
                     Clip prediction to [1, 5]
```

### Components and rationale

- **Canonical Whisper transcript:** Provides normalized text for contextual grammar analysis.
- **Verbatim Whisper transcript:** Retains more spoken wording and disfluencies, providing a different view of the utterance.
- **Wav2Vec2 CTC transcript:** Adds a transcription pathway with different recognition characteristics.
- **DeBERTa regressors:** Model contextual language patterns and output continuous grammar-score predictions.
- **WavLM embeddings + RBF-SVR:** Add an acoustic signal that does not depend solely on recognized text.
- **Whisper embeddings + RBF-SVR:** Provide a second learned audio representation and nonlinear regression.
- **Non-negative linear stack:** Combines base-model predictions with non-negative learned coefficients.
- **Score clipping:** Constrains final predictions to the intended non-zero rubric range of 1 to 5.

## Workflow

1. **Exploratory data analysis**
   - Inspect the train and test datasets.
   - Examine the target distribution and zero-label rows.
   - Validate array shapes, row counts, and prediction values.
2. **Model development and experimentation**
   - Compare transcript-based and audio-based predictions.
   - Reuse saved predictions and embeddings where available.
3. **Evaluation**
   - Calculate RMSE, MAE, Pearson correlation, and \(R^2\).
   - Plot predicted versus actual scores.
   - Inspect residual distributions and error by true score.
4. **Ensemble evaluation**
   - Evaluate a non-negative linear stack using cross-validation over saved out-of-fold (OOF) predictions.
   - Inspect the fitted stacking coefficients.
5. **Test inference**
   - Combine test predictions from the base models.
   - Validate the final output and write `submission.csv`.

## Dataset

The competition data includes:

- `train.csv`: training filenames and target labels.
- `test.csv`: filenames requiring predictions.
- Audio files for the training and test sets.

The working data used in the experiments contained **769 training rows** and **216 test rows**. The main regression evaluation used **732 rows with labels greater than zero**, excluding 37 zero-label rows to match the saved artifacts.

The competition data and audio files are not included in this repository. Access them through the official competition and comply with its terms.

## Evaluation Results

The latest reported cross-validated evaluation of the saved-prediction stack is shown below.

| Model | RMSE ↓ | MAE ↓ | Pearson r ↑ | R² ↑ |
|---|---:|---:|---:|---:|
| **Non-negative linear stack** | **0.495771** | **0.381186** | **0.872509** | **0.760822** |
| Whisper | 0.583209 | 0.459272 | 0.820697 | 0.669016 |
| WavLM | 0.591130 | 0.451920 | 0.813261 | 0.659964 |
| CTC | 0.605535 | 0.469649 | 0.812475 | 0.643190 |
| Canonical | 0.616168 | 0.488323 | 0.797048 | 0.630548 |
| Verbatim | 0.621969 | 0.487574 | 0.794867 | 0.623560 |

**Interpretation:** The stack has the lowest RMSE and MAE among the listed models. Its RMSE is approximately **15.0% lower** than the best individual model by RMSE, Whisper. The Pearson correlation indicates a strong positive relationship between predicted and true scores.

### Leaderboard reference

The previous submitted file achieved a **public leaderboard score of 0.3412**. This is a historical competition result, not the cross-validation result above. A new submission's score must be checked on the competition leaderboard; improvement is not guaranteed.

### Evaluation caveat

The stack metrics are calculated by cross-validating a second-stage regressor over saved base-model OOF predictions. The base predictions were produced in prior experiments, and their fold compatibility has not been independently verified end to end. Therefore, these metrics are useful for model comparison but should not be interpreted as a fully independent final generalization estimate.

## Metrics

- **RMSE:** Penalizes larger prediction errors more heavily.
- **MAE:** Measures average absolute prediction error.
- **Pearson correlation:** Measures linear association between predictions and labels.
- **R²:** Measures explained variance relative to a mean-prediction baseline.

## Getting Started

### Clone the repository

```bash
git clone <your-repository-url>
cd Grammar_Scoring
```

### Prepare the environment

The notebooks use Python with common data-science and machine-learning libraries, including:

- NumPy
- pandas
- Matplotlib
- scikit-learn

The original model-development notebook may require additional libraries for transformer models, speech processing, and embedding extraction, depending on which training cells are executed.

### Provide the required data and artifacts

Make the following available in the notebook environment:

- Competition train/test CSV files and audio files.
- Saved OOF and test prediction arrays from the previous model runs.
- Saved embeddings required by the Whisper audio-regression branch.

The notebooks were developed using Kaggle-style paths. Update the path configuration cells if running elsewhere. The clean notebook reuses saved artifacts; it does not recreate missing neural-model outputs automatically.

### Generate predictions

Run `Grammar_Scoring_Engine.ipynb` from top to bottom. The final submission is saved at:

```text
/kaggle/working/grammar_scoring_final/submission.csv
```

The file contains the test filenames and predicted `label` values.

## Repository Structure

```text
Grammar_Scoring/
├── README.md
├── Grammar_Scoring_Experiments.ipynb
└── Grammar_Scoring_Engine.ipynb
```
