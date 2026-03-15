# Final Report Outline

This outline is tailored to the `tab-cnn` project and organized to match the standard CS229 final report flow used in the provided template: title/authors, abstract, introduction, related work, dataset/features, methods, experiments/results, conclusion, and references.

Because I could not reliably extract the PDF's exact heading text in this environment, I mirrored the standard CS229 report structure and filled each section with project-specific content from this repository.

The outline is intended to cover the project arc end-to-end: initial reproduction, early training regressions, the first bidirectional-GRU CRNN experiments, later simplification to a smaller unidirectional GRU, and the final CNN-versus-CRNN comparison after the training setup stabilized.

## Title And Metadata

Use a title that makes the contribution clear, for example:

**Guitar Tablature Estimation from Audio with CNN and CRNN Models**

Include:

- Project title
- Team member names and SUNet IDs
- Course and term
- Link to code repository if allowed by the template

## Abstract

Keep this to one short paragraph.

Include:

- The task: frame-level guitar tablature estimation from audio
- The motivation: automatic transcription is useful but tablature prediction is harder than note transcription because it must infer string and fret jointly
- The approach: reproduce and extend TabCNN on GuitarSet using CQT inputs, diagnose early failure modes, and compare several CNN/CRNN training configurations
- The project arc:
  - initial reproduction and modernization
  - early class-collapse and under-prediction failures
  - first CRNN experiments with a larger bidirectional GRU
  - later simplification to a smaller unidirectional GRU with auxiliary sounding loss
- The main result: final stabilized runs show CNN and CRNN are close overall, with CRNN improving pitch-level F1 while CNN remains stronger on some tablature metrics
- A brief conclusion: temporal modeling helps some metrics, but gains are mixed rather than dominant

Concrete numbers you can cite:

- Best recent CNN run: `pf = 0.8191`, `tf = 0.7452`, `tdr = 0.9040` from [`model/saved/c_cnn 2026-03-09 09-57-36/results.csv`](/home/arminhaj/projects/tab-cnn/model/saved/c_cnn%202026-03-09%2009-57-36/results.csv)
- Best recent CRNN run by pitch-F1: `pf = 0.8416`, `tf = 0.7437` from [`model/saved/c_crnn 2026-03-12 19-47-38/results.csv`](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-12%2019-47-38/results.csv)

## 1. Introduction

Explain the problem and why it matters.

Include:

- What guitar tablature estimation is
- Why it is difficult: polyphony, timing ambiguity, multiple possible fingerings for the same pitch, noise and expressive playing
- Why GuitarSet is a good benchmark
- What this project studies:
  - reproduction of the original TabCNN setup
  - modernization of the training pipeline
  - diagnosis of early failure modes during training
  - comparison between a pure CNN and multiple CNN+RNN variants
- Your high-level research question:
  - Does adding temporal recurrence improve frame-level tablature estimation over a convolution-only baseline?

End the introduction with a short summary of contributions, for example:

- Reproduced the TabCNN pipeline on GuitarSet
- Implemented and evaluated a CRNN variant
- Evaluated an initial bidirectional-GRU design and later simplified it
- Analyzed regressions caused by training-pipeline changes
- Compared models across cross-validation folds using standard tablature metrics

## 2. Related Work

Summarize prior work briefly and position this project relative to it.

Include:

- The original TabCNN paper/repository that this codebase is based on
- Prior work in automatic music transcription and guitar-specific transcription
- Why tablature estimation is different from generic multi-pitch estimation
- Why CNNs are natural for local time-frequency pattern extraction
- Why CRNNs might help by modeling temporal continuity across neighboring frames

State clearly:

- This project is not proposing a fundamentally new task
- The contribution is empirical: reproduction, modernization, comparison, and analysis

## 3. Dataset And Features

Describe the data and preprocessing pipeline.

Include:

- Dataset: GuitarSet
- Data location and preprocessing scripts:
  - [`data/TabDataReprGen.py`](/home/arminhaj/projects/tab-cnn/data/TabDataReprGen.py)
  - [`data/Parallel_TabDataReprGen.py`](/home/arminhaj/projects/tab-cnn/data/Parallel_TabDataReprGen.py)
- Input representation used in your main experiments: CQT (`spec_repr = "c"`)
- Context window size: `con_win_size = 9`
- Output formulation:
  - 6 guitar strings
  - 21 classes per string
  - class `0` or `M` corresponds to muted/no note, remaining classes correspond to fret positions

Describe the split strategy from the code:

- Cross-validation by held-out guitarist identity
- Within the held-out guitarist, a validation/test split is created by track
- This tests generalization across players rather than only across clips

Point to the implementation:

- [`model/TabCNN.py`](/home/arminhaj/projects/tab-cnn/model/TabCNN.py)

Useful figure/table ideas:

- A table summarizing dataset size, number of players, and train/validation/test strategy
- A figure showing one example CQT frame window and its corresponding tablature label

## 4. Methods

Split this section into baseline, CRNN extension, training setup, and metrics. Make sure it captures both the initial and final CRNN designs rather than only the final stabilized configuration.

### 4.1 Baseline CNN

Describe the baseline architecture shown in the run log.

Include:

- Input shape for CQT: `(192, 9, 1)`
- Three convolution layers followed by pooling and dropout
- Flattening and dense layers
- Final output reshaped to `(6, 21)` with per-string softmax
- Rough parameter count: about `834k`

Support with:

- [`model/saved/c_cnn 2026-03-09 09-57-36/log.txt`](/home/arminhaj/projects/tab-cnn/model/saved/c_cnn%202026-03-09%2009-57-36/log.txt)

### 4.2 CRNN Variant

Describe the recurrent extension as an evolving design, not a single fixed model.

Include:

- Convolutional frontend for feature extraction
- Temporal sequence modeling using a GRU
- Initial CRNN configuration:
  - `rnn_type = "gru"`
  - `rnn_units = 128`
  - `rnn_layers = 1`
  - `bidirectional = True`
  - parameter count roughly `2.56M`
- Later stabilized configuration:
  - `rnn_type = "gru"`
  - `rnn_units = 64`
  - `rnn_layers = 1`
  - `bidirectional = False`
- Dense output head predicting 6 string-wise fret distributions
- Rough parameter count of final version: about `683k`

Support with:

- [`model/saved/c_crnn 2026-03-09 22-18-24/log.txt`](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-09%2022-18-24/log.txt)
- [`model/saved/c_crnn 2026-03-12 07-56-47/log.txt`](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-12%2007-56-47/log.txt)

### 4.3 Training Setup

Report both the common training setup and the important configuration changes across the project timeline.

Include:

- Batch size: `128`
- Epochs: `8`
- Optimizer learning rate: `1.0` as logged
- GPU training with TensorFlow backend
- Held-out validation fraction: `0.2`
- No early stopping in the main logged runs
- Whether class weighting and auxiliary losses were enabled for each phase of the project
- Which runs used bidirectionality, mixed precision, or the sounding auxiliary loss

Be explicit about differences between models:

- Early bad runs were associated with training-pipeline changes and severe under-prediction
- Early CRNN runs used a larger bidirectional GRU
- Later CRNN runs switched to a smaller unidirectional GRU and introduced sounding-aware supervision
- CNN final run used no sounding auxiliary loss

### 4.4 Evaluation Metrics

Define the metrics used in `results.csv`.

Include:

- `pp`, `pr`, `pf`: pitch precision, recall, F1
- `tp`, `tr`, `tf`: tablature precision, recall, F1
- `tdr`: tablature disambiguation rate
- `ind`: some combined or index-style summary metric used by the repository

You should verify the exact metric definitions from:

- [`model/Metrics.py`](/home/arminhaj/projects/tab-cnn/model/Metrics.py)

This section should explain why both pitch and tablature metrics matter:

- A model can predict the correct pitch while assigning the wrong string/fret position
- Tablature metrics are therefore stricter and more musically meaningful for this task

## 5. Experiments

This section should state the experimental questions clearly and follow the actual chronology of the project rather than jumping directly to the best final runs.

Recommended subsections:

### 5.1 Reproduction Of Original TabCNN

Include:

- Whether the reimplementation reproduces the expected behavior of the original project
- Any implementation changes needed for modern Keras/TensorFlow
- Whether modernized runs initially regressed

### 5.2 Early Failure Mode: Under-Prediction And Class Collapse

Include:

- The first major problem: both CNN and CRNN heavily over-predicted class `0`
- Why this mattered: high precision but extremely poor recall and unrealistic empty-string outputs
- How this was diagnosed from predictions and class frequencies
- The likely cause: severe class imbalance combined with training-stack changes

Support with:

- [`model/saved/comparisons/cnn_crnn_prediction_collapse_writeup.md`](/home/arminhaj/projects/tab-cnn/model/saved/comparisons/cnn_crnn_prediction_collapse_writeup.md)
- [`model/saved/comparisons/regression_analysis_vs_original_tabcnn.md`](/home/arminhaj/projects/tab-cnn/model/saved/comparisons/regression_analysis_vs_original_tabcnn.md)

### 5.3 Initial CRNN Design: Bidirectional GRU

Include:

- The first serious CRNN experiments used a larger bidirectional GRU
- This increased model capacity substantially relative to the final CRNN
- Describe the empirical outcome:
  - stronger pitch-level results than the collapsed early runs
  - but not a clean, decisive win on tablature metrics
  - signs that the larger recurrent model was harder to regularize and less attractive than a simpler alternative
- If you observed overfitting in training curves or validation behavior, describe it here explicitly

Support with:

- [`model/saved/c_crnn 2026-03-09 22-18-24/log.txt`](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-09%2022-18-24/log.txt)
- [`model/saved/c_crnn 2026-03-09 22-18-24/results.csv`](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-09%2022-18-24/results.csv)

### 5.4 Simplified CRNN And Final CNN vs CRNN Comparison

This should be the main stabilized comparison, but it should be presented as the endpoint of the earlier iteration cycle.

Include:

- Same input representation and comparable training settings for both models
- Cross-fold evaluation over guitarist splits
- Comparison of mean and standard deviation across folds

Use the comparison summary from:

- [`model/saved/comparisons/cnn_new_vs_crnn_new_summary.md`](/home/arminhaj/projects/tab-cnn/model/saved/comparisons/cnn_new_vs_crnn_new_summary.md)

Key points already supported by repo artifacts:

- CRNN improves pitch recall and pitch F1 relative to the final CNN comparison
- CNN remains stronger on tablature precision, tablature F1, and TDR in that comparison
- The result is a tradeoff, not a universal CRNN win

Concrete comparison numbers to include:

- CNN: `pf = 0.8191`, `tf = 0.7452`, `tdr = 0.9040`
- CRNN: `pf = 0.8416`, `tf = 0.7437`, `tdr = 0.8718`

### 5.5 Training Stability / Regression Analysis

This is worth including because the repo contains explicit evidence for it.

Include:

- Earlier runs collapsed toward under-prediction or dominant mute-class behavior
- Training-stack changes likely contributed to this regression
- Important changes:
  - move from older generator training to `tf.data`
  - mixed precision in some bad runs
  - optimization/runtime refactors

Support with:

- [`model/saved/comparisons/regression_analysis_vs_original_tabcnn.md`](/home/arminhaj/projects/tab-cnn/model/saved/comparisons/regression_analysis_vs_original_tabcnn.md)

This section strengthens the report because it shows debugging and scientific iteration rather than only final numbers.

## 6. Results

Make this section data-heavy and concise, but do not present only the final snapshot. Summarize the trajectory from failure, to initial CRNN success, to the final stabilized comparison.

Include at least one summary table:

- Mean and standard deviation across folds for CNN and CRNN
- Emphasize `pf`, `tf`, and `tdr`
- A second table or short paragraph summarizing the progression of CRNN variants:
  - collapsed early run
  - bidirectional-GRU run
  - final unidirectional-GRU run

Useful progression numbers already in the repo:

- Collapsed CRNN run (`2026-02-23`): `pf = 0.1542`, `tf = 0.1261`
- Stronger bidirectional-GRU CRNN run (`2026-03-09`): `pf = 0.8289`, `tf = 0.7403`
- Final unidirectional-GRU CRNN run (`2026-03-12`): `pf = 0.8416`, `tf = 0.7437`

Suggested interpretation of current results:

- CRNN produces the best pitch-F1 among the recent runs
- CNN still has slightly better tablature-F1 and substantially better TDR in the latest comparison
- If the objective is pitch-level transcription, CRNN looks favorable
- If the objective is exact fingering/tab prediction, the CNN remains competitive or preferable

Useful visualizations already available in the repo:

- Loss curves and average accuracy curves under the fold directories
- Demo videos/GIFs in CRNN saved runs for qualitative examples

Good figure ideas:

- Bar chart comparing CNN vs CRNN on `pf`, `tf`, and `tdr`
- Per-fold scatter plot or grouped bar chart to show variance across guitarists
- One qualitative prediction visualization where the model gets pitch right but fingering wrong

## 7. Discussion

This section should interpret the results and address limitations.

Include:

- Why temporal recurrence likely helps pitch recall:
  - neighboring frames are correlated
  - note sustain and transitions are temporally smooth
- Why the first bidirectional-GRU design may not have been the final answer:
  - it was much larger than the simplified CRNN
  - it was harder to justify if gains over the simpler GRU were small or unstable
- Why recurrence does not automatically improve tablature metrics:
  - exact fingering remains ambiguous even when pitch is correct
  - the model may smooth over local fingering distinctions
- The impact of class imbalance, especially the mute/no-note class
- The sensitivity of training behavior to implementation/runtime details

Discuss limitations honestly:

- Small dataset relative to modern deep learning standards
- Limited hyperparameter search
- Only one main spectral representation emphasized in final experiments
- Evaluation remains frame-based rather than phrase- or note-level

## 8. Conclusion And Future Work

Summarize the main takeaway in a few sentences.

Recommended conclusion:

- The project reproduced a workable guitar tablature estimation pipeline on GuitarSet
- The project moved through several phases: reproduction, collapse diagnosis, bidirectional-GRU experiments, and a final simplified CRNN comparison
- Adding recurrence improved pitch-level performance but did not consistently improve stricter tablature metrics
- Final model choice depends on whether pitch accuracy or fingering accuracy is the priority

Future work ideas:

- Tune class weighting and auxiliary losses more systematically
- Compare CQT against mel, STFT, and concatenated inputs
- Add note-level temporal decoding instead of framewise prediction only
- Explore bidirectional recurrent layers or transformer-style temporal modeling
- Calibrate losses specifically for tablature accuracy rather than only pitch recovery

## 9. Individual Contributions

If the template includes this section, list concrete ownership.

Include:

- Data preprocessing
- Model implementation / modernization
- Experiment management
- Metric analysis and visualization
- Writing and editing responsibilities

## 10. References

Include at least:

- The original TabCNN paper
- The GuitarSet dataset paper
- Any additional transcription or sequence-modeling papers cited in related work

## Suggested Tables And Figures Checklist

Use this as a short production checklist while writing.

- Table: dataset and split summary
- Table: CNN vs CRNN metrics across folds
- Figure: model architecture diagram or concise block diagram
- Figure: example CQT input and target tablature
- Figure: training curves for one representative fold
- Figure: qualitative prediction example from demo outputs

## Recommended Narrative Arc

If you want the report to read cleanly, frame it like this:

1. Tablature estimation is harder than ordinary pitch transcription because fingering matters.
2. A CNN baseline is strong at local spectral pattern recognition.
3. A CRNN may exploit temporal continuity and improve transcription quality.
4. In practice, the project first encountered collapse and instability before reaching a stable CRNN comparison.
5. The initial bidirectional-GRU CRNN was part of that iteration path, but a smaller unidirectional GRU became the cleaner final comparison point.
6. Training dynamics and implementation details materially affect results, so careful reproduction matters.
