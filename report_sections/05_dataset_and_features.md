# Dataset And Features

The experiments use GuitarSet, a dataset of recorded solo guitar performances paired with aligned annotation data. Within this repository, preprocessing is handled by [TabDataReprGen.py](/home/arminhaj/projects/tab-cnn/data/TabDataReprGen.py) and [Parallel_TabDataReprGen.py](/home/arminhaj/projects/tab-cnn/data/Parallel_TabDataReprGen.py), which transform the raw audio into compressed spectral representations stored under `data/spec_repr/`.

The primary input representation in the final experiments is the constant-Q transform, enabled with `spec_repr = "c"`. For this setting, each training example is a local spectral window with shape `(192, 9, 1)`, where 192 is the frequency dimension and 9 frames provide short temporal context. The output is a six-string tablature target. Each string is assigned one of 21 classes: class 0 denotes a non-sounding or muted state, and classes 1 through 20 correspond to playable fret positions.

The split design is important. Rather than mixing all recordings randomly, the code partitions data by guitarist identity and evaluates with a held-out guitarist in each fold. Within that held-out set, tracks are split into validation and test subsets. This design makes the problem more realistic because the model is judged on players not seen during training.

The label design also explains why pitch and tablature metrics can diverge. A single predicted frame contains six per-string decisions, and those decisions map to both a set of pitches and a specific fingering configuration. As a result, a model can be correct at the pitch level while still assigning notes to the wrong strings or frets.

One useful figure for this section would be an example input-target pairing. The repository does not currently include a static CQT-plus-label figure, but the preprocessing code and model input specification are defined in [TabCNN.py](/home/arminhaj/projects/tab-cnn/model/TabCNN.py), which is the right source if you want to generate one later.
