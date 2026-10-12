# Audio Grammar Scoring

**Spoken-English Grammar Score Prediction**

A machine-learning solution for predicting a continuous grammar score
from spoken-English recordings. The project combines transcript-based
language models with audio representations and regression models, then
evaluates an ensemble using out-of-fold (OOF) predictions.

## Problem Statement

Given an audio recording of a person speaking English, predict its
grammar score. The training data contains `filename` and `label`; the
test data contains `filename` only. The final submission must contain
the test filenames and their predicted labels.

The available dataset has 769 training rows and 216 test rows. In the
experiments, rows with a zero label were excluded from the eligible
supervised evaluation/training population, leaving 732 rows. Confirm
this filtering rule against the competition rubric before reusing the
pipeline on a different dataset.

## Model Architecture

The final ensemble uses five complementary prediction streams:

1.  **Wav2Vec2 CTC transcript → DeBERTa regressor** --- uses a
    CTC-generated transcript.
2.  **Canonical Whisper transcript → DeBERTa regressor** --- uses a
    normalized Whisper transcript.
3.  **Verbatim Whisper transcript → DeBERTa regressor** --- preserves a
    more speech-faithful transcript variant.
4.  **WavLM audio embeddings → SVR** --- predicts from audio embeddings
    using a scaled support-vector regressor.
5.  **Whisper audio embeddings → RBF-SVR** --- predicts from Whisper
    audio embeddings using an RBF-kernel support-vector regressor.

The five base predictions are combined by a **non-negative linear
stacker**. A **linear calibration** stage then adjusts the stacked
predictions' scale and offset.

``` text
                         Speech recordings
                                 |
              +------------------+------------------+
              |                  |                  |
        Wav2Vec2 CTC       Whisper canonical   Whisper verbatim
         transcript           transcript          transcript
              |                  |                  |
              +------------------+------------------+
                                 |
                       DeBERTa regressors
                                 |
                       Three text predictions
                                 |
                                 +-----------------------------+
                                                               |
        Audio recordings                                       |
              |                                                |
        +-----+-------------------+                            |
        |                         |                            |
   WavLM embeddings        Whisper embeddings                  |
        |                         |                            |
   Scaled SVR                 RBF-SVR                          |
        |                         |                            |
        +-------------------------+----------------------------+
                                 |
                       Five prediction streams
                                 |
                   Non-negative linear stacking
                                 |
                       Linear calibration
                                 |
                     Predicted grammar scores
                                 |
                       Submission CSV file
```

## Repository / Notebook Workflow

The project is organized around two workflows:

-   **Experimentation:** data exploration, feature/model experiments,
    OOF generation, model comparisons, and ensemble selection.
-   **Evaluation and inference:** loads saved predictions and
    embeddings, checks alignment, evaluates base models and ensembles,
    performs calibration, and creates a submission.

The final inference notebook is designed to reuse saved artifacts. It
does **not** retrain the expensive neural models or repeat audio feature
extraction. It still requires the relevant saved OOF predictions and
embeddings to be attached to the Kaggle notebook as inputs, or available
at the configured paths.

## Evaluation

Evaluation is performed on eligible training rows using out-of-fold
predictions so that each row's evaluation prediction is produced without
fitting the corresponding validation fold. The notebook reports
regression metrics and diagnostic plots, including model-level
performance, ensemble comparisons, residual/error analysis, and
prediction-versus-target visualizations.

The exact evaluation results depend on the artifacts loaded and the
cross-validation/calibration procedure used.

----------------------------------------------------------------------------
  Model                OOF RMSE ↓      OOF MAE ↓        Pearson           R² ↑
                                                  correlation ↑ 
  ---------------- -------------- -------------- -------------- --------------
  **Non-negative     **0.495771**   **0.381186**   **0.872509**   **0.760822**
  linear stack**                                                

  Whisper RBF-SVR        0.583209       0.459272       0.820697       0.669016

  WavLM SVR              0.591130       0.451920       0.813261       0.659964

  CTC + DeBERTa          0.605535       0.469649       0.812475       0.643190

  Canonical              0.616168       0.488323       0.797048       0.630548
  Whisper +                                                     
  DeBERTa                                                       

  Verbatim               0.621969       0.487574       0.794867       0.623560
  Whisper +                                                     
  DeBERTa                                                       
  ----------------------------------------------------------------------------

The stack had the lowest RMSE and MAE among these evaluated options. Its
RMSE was approximately **15% lower than the best individual model's
RMSE** in this comparison. This supports the finding that the different
prediction streams contribute complementary information: the ensemble
performed better locally than any one of its constituent prediction
streams.

The audio-based models were the strongest individual models in this
comparison, with Whisper RBF-SVR leading on RMSE and WavLM SVR close
behind. The three transcript-based DeBERTa predictors each provided a
distinct text signal, but none outperformed the two audio regressors
individually on the reported metrics.


## Conclusion

The strongest reported solution combines three transcript-based DeBERTa
predictors with two audio-embedding regressors, uses non-negative linear
stacking to combine their outputs, and applies linear calibration before
submission. In the saved-prediction evaluation, the stack outperformed
each individual predictor. The best reported public leaderboard
submission, `predictions_stack_linear_calibrated.csv`, achieved
**0.3344**.
