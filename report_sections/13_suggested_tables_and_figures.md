# Suggested Tables And Figures

The most useful summary table is the CNN-versus-CRNN metric comparison in [cnn_new_vs_crnn_new_summary.md](/home/arminhaj/projects/tab-cnn/model/saved/comparisons/cnn_new_vs_crnn_new_summary.md). It already aggregates mean and standard deviation across folds and can be converted directly into a report table. A second small table should summarize the project trajectory with one row each for the collapsed early CRNN, the bidirectional-GRU CRNN, and the final simplified CRNN.

Recommended figure links:

- CNN training curve example: [loss_curve.png](/home/arminhaj/projects/tab-cnn/model/saved/c_cnn%202026-03-09%2009-57-36/0/loss_curve.png)
- CRNN training curve example: [loss_curve.png](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-12%2019-47-38/0/plots/loss_curve.png)
- CNN accuracy curve example: [avg_acc_curve.png](/home/arminhaj/projects/tab-cnn/model/saved/c_cnn%202026-03-09%2009-57-36/0/avg_acc_curve.png)
- CRNN accuracy curve example: [avg_acc_curve.png](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-12%2019-47-38/0/plots/avg_acc_curve.png)
- Validation confusion matrix example: [val_fret_confusion_epoch_08.png](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-12%2019-47-38/0/confusion/png/val_fret_confusion_epoch_08.png)
- Qualitative tablature demo example: [c_crnn 2026-03-08 17-07-40_fold0_00_BN1-129-Eb_comp.gif](/home/arminhaj/projects/tab-cnn/model/saved/c_crnn%202026-03-08%2017-07-40/0/demos/c_crnn%202026-03-08%2017-07-40_fold0_00_BN1-129-Eb_comp.gif)
- Early collapse diagnosis writeup: [cnn_crnn_prediction_collapse_writeup.md](/home/arminhaj/projects/tab-cnn/model/saved/comparisons/cnn_crnn_prediction_collapse_writeup.md)

If you need a single comparison table in CSV form, use [cnn_new_vs_crnn_new_summary.csv](/home/arminhaj/projects/tab-cnn/model/saved/comparisons/cnn_new_vs_crnn_new_summary.csv). If you need per-fold numbers for a variance plot, use [cnn_new_vs_crnn_new_per_fold.csv](/home/arminhaj/projects/tab-cnn/model/saved/comparisons/cnn_new_vs_crnn_new_per_fold.csv).
