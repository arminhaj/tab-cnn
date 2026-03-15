''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
'''

from __future__ import print_function
import argparse
import csv
import keras
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from pathlib import Path
try:
    import tensorflow as tf
except ImportError:
    tf = None
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Permute, TimeDistributed
from keras.layers import GRU, LSTM, Bidirectional
from DataGenerator import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *

def save_confusion_heatmap(confusion, output_path, title, num_classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    positive_confusion = np.ma.masked_less_equal(confusion, 0)
    if positive_confusion.count() > 0:
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color="black")
        image = ax.imshow(
            positive_confusion,
            cmap=cmap,
            aspect="auto",
            norm=LogNorm(vmin=1, vmax=positive_confusion.max()),
        )
        colorbar_label = "Count (log scale)"
    else:
        image = ax.imshow(confusion, cmap="magma", aspect="auto")
        colorbar_label = "Count"
    ax.set_xlabel("Predicted fret class")
    ax.set_ylabel("True fret class")
    ax.set_title(title)
    tick_positions = np.arange(num_classes)
    tick_labels = ["M"] + [str(idx) for idx in range(num_classes - 1)]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

class TabCNN:
    
    def __init__(self, 
                 batch_size=128, 
                 epochs=8,
                 con_win_size = 9,
                 spec_repr="c",
                 data_path=None,
                 id_file="id.csv",
                 save_path=None,
                 architecture="crnn",
                 rnn_type="gru",
                 rnn_units=128,
                 rnn_layers=1,
                 bidirectional=True,
                 rnn_recurrent_dropout=0.2,
                 use_early_stopping=False,
                 early_stopping_patience=2,
                 heldout_val_fraction=0.2,
                 heldout_split_seed=1337,
                 generator_cache_files=32,
                 use_tf_data=True,
                 enable_mixed_precision=False,
                 enable_xla=False,
                 gpu_memory_growth=True,
                 use_class_weighting=False,
                 class_weight_power=0.5,
                 min_class_weight=0.1,
                 max_class_weight=8.0,
                 closed_class_weight=0.25,
                 use_sounding_aux_loss=False,
                 sounding_aux_weight=0.5,
                 sounding_positive_weight=2.0,
                 optimizer_learning_rate=1.0,
                 track_validation_diagnostics=True,
                 checkpoint_validation_metrics=True):   
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.architecture = architecture.lower()
        self.rnn_type = rnn_type.lower()
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.rnn_recurrent_dropout = rnn_recurrent_dropout
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.heldout_val_fraction = heldout_val_fraction
        self.heldout_split_seed = heldout_split_seed
        self.generator_cache_files = generator_cache_files
        self.use_tf_data = use_tf_data
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_xla = enable_xla
        self.gpu_memory_growth = gpu_memory_growth
        self.use_class_weighting = use_class_weighting
        self.class_weight_power = class_weight_power
        self.min_class_weight = min_class_weight
        self.max_class_weight = max_class_weight
        self.closed_class_weight = (
            None if closed_class_weight is None else float(closed_class_weight)
        )
        if use_sounding_aux_loss is None:
            self.use_sounding_aux_loss = (self.architecture == "crnn")
        else:
            self.use_sounding_aux_loss = bool(use_sounding_aux_loss)
        self.sounding_aux_weight = sounding_aux_weight
        self.sounding_positive_weight = sounding_positive_weight
        self.optimizer_learning_rate = optimizer_learning_rate
        self.track_validation_diagnostics = bool(track_validation_diagnostics)
        self.checkpoint_validation_metrics = bool(checkpoint_validation_metrics)
        self._tf_backend = False
        self._gpu_devices = []

        if self.architecture not in {"cnn", "crnn"}:
            raise ValueError("architecture must be 'cnn' or 'crnn'")
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        if self.architecture == "crnn" and self.rnn_layers < 1:
            raise ValueError("rnn_layers must be >= 1 when using architecture='crnn'")
        if not (0.0 < self.heldout_val_fraction < 1.0):
            raise ValueError("heldout_val_fraction must be in the open interval (0, 1)")

        model_dir = Path(__file__).resolve().parent
        project_root = model_dir.parent
        default_data_path = project_root / "data" / "spec_repr"
        default_save_path = model_dir / "saved"
        self.data_path = Path(data_path) if data_path is not None else default_data_path
        self.id_file = id_file
        self.save_path = Path(save_path) if save_path is not None else default_save_path

        self.configure_runtime()
        
        self.load_IDs()
        
        run_name = f"{self.spec_repr}_{self.architecture} " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.save_folder = self.save_path / run_name
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_folder / "log.txt"
        
        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["ind"] = []
        self.metrics["data"] = ["g0","g1","g2","g3","g4","g5","mean","std dev"]
        
        if self.spec_repr == "c":
            self.input_shape = (192, self.con_win_size, 1)
        elif self.spec_repr == "m":
            self.input_shape = (128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.input_shape = (320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.input_shape = (1025, self.con_win_size, 1)
            
        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6
        self.class_weights = np.ones(self.num_classes, dtype=np.float32)

    def load_IDs(self):
        csv_file = self.data_path / self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])

    def configure_runtime(self):
        if tf is None:
            return

        try:
            self._tf_backend = (keras.backend.backend() == "tensorflow")
        except Exception:
            self._tf_backend = False

        if not self._tf_backend:
            return

        self._gpu_devices = tf.config.list_physical_devices("GPU")

        if self.gpu_memory_growth:
            for gpu in self._gpu_devices:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass

        if self.enable_xla:
            try:
                tf.config.optimizer.set_jit(True)
            except Exception:
                pass

        if self.enable_mixed_precision and self._gpu_devices:
            try:
                keras.mixed_precision.set_global_policy("mixed_float16")
            except Exception:
                pass

    def generator_data_path(self):
        return str(self.data_path) + "/"
        
    def partition_data(self, data_split):
        self.data_split = data_split
        self.partition = {}
        self.partition["training"] = []
        self.partition["heldout"] = []
        for ID in self.list_IDs:
            guitarist = int(ID.split("_")[0])
            if guitarist == data_split:
                self.partition["heldout"].append(ID)
            else:
                self.partition["training"].append(ID)

        self.partition["validation"], self.partition["test"] = self._split_heldout_by_track(
            self.partition["heldout"],
            self.heldout_val_fraction,
            self.heldout_split_seed + int(data_split),
        )
                
        self.training_generator = DataGenerator(self.partition['training'], 
                                                data_path=self.generator_data_path(), 
                                                batch_size=self.batch_size, 
                                                shuffle=True,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size,
                                                cache_files=self.generator_cache_files)

        # Validation during fit() should be mini-batched to avoid OOM.
        validation_fit_batch_size = min(self.batch_size, len(self.partition["validation"]))
        self.validation_fit_generator = DataGenerator(
            self.partition["validation"],
            data_path=self.generator_data_path(),
            batch_size=validation_fit_batch_size,
            shuffle=False,
            spec_repr=self.spec_repr,
            con_win_size=self.con_win_size,
            cache_files=self.generator_cache_files,
        )

        self.validation_eval_generator = DataGenerator(
            self.partition["validation"],
            data_path=self.generator_data_path(),
            batch_size=len(self.partition["validation"]),
            shuffle=False,
            spec_repr=self.spec_repr,
            con_win_size=self.con_win_size,
            cache_files=self.generator_cache_files,
        )

        # Keep full-batch test set for post-training prediction/evaluation.
        self.test_generator = DataGenerator(
            self.partition["test"],
            data_path=self.generator_data_path(),
            batch_size=len(self.partition["test"]),
            shuffle=False,
            spec_repr=self.spec_repr,
            con_win_size=self.con_win_size,
            cache_files=self.generator_cache_files,
        )
        
        self.split_folder = self.save_folder / str(self.data_split)
        self.split_folder.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.split_folder / "metrics"
        self.plots_dir = self.split_folder / "plots"
        self.confusion_csv_dir = self.split_folder / "confusion" / "csv"
        self.confusion_plot_dir = self.split_folder / "confusion" / "png"
        self.checkpoints_dir = self.split_folder / "checkpoints"
        self.artifacts_dir = self.split_folder / "artifacts"
        for directory in (
            self.metrics_dir,
            self.plots_dir,
            self.confusion_csv_dir,
            self.confusion_plot_dir,
            self.checkpoints_dir,
            self.artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.update_class_weights()

    def _split_heldout_by_track(self, heldout_ids, val_fraction, seed):
        # Keep all frames from a track together to avoid leakage.
        track_to_ids = {}
        for sample_id in heldout_ids:
            track_key = "_".join(sample_id.split("_")[:-1])
            track_to_ids.setdefault(track_key, []).append(sample_id)

        track_keys = np.array(sorted(track_to_ids.keys()))
        if len(track_keys) < 2:
            raise ValueError("Need at least 2 held-out tracks to create validation/test split.")

        rng = np.random.default_rng(seed)
        rng.shuffle(track_keys)
        val_track_count = int(round(len(track_keys) * val_fraction))
        val_track_count = max(1, min(len(track_keys) - 1, val_track_count))

        val_tracks = set(track_keys[:val_track_count].tolist())
        validation_ids = []
        test_ids = []
        for track_key, ids in track_to_ids.items():
            if track_key in val_tracks:
                validation_ids.extend(ids)
            else:
                test_ids.extend(ids)
        return validation_ids, test_ids

    def update_class_weights(self):
        weights = np.ones(self.num_classes, dtype=np.float64)

        if self.use_class_weighting:
            data_dir = self.data_path / self.spec_repr
            track_stems = sorted({"_".join(ID.split("_")[:-1]) for ID in self.partition["training"]})
            class_counts = np.zeros(self.num_classes, dtype=np.float64)

            for track in track_stems:
                npz_path = data_dir / f"{track}.npz"
                with np.load(npz_path, allow_pickle=False) as loaded:
                    labels = loaded["labels"]
                class_counts += labels.sum(axis=(0, 1))

            seen = class_counts > 0
            inv_freq = np.ones(self.num_classes, dtype=np.float64)
            mean_seen_count = np.mean(class_counts[seen])
            inv_freq[seen] = mean_seen_count / class_counts[seen]
            weights = np.power(inv_freq, self.class_weight_power)
            weights /= np.mean(weights[seen])
            weights = np.clip(weights, self.min_class_weight, self.max_class_weight)

        if self.closed_class_weight is not None:
            weights[0] = self.closed_class_weight

        self.class_weights = weights.astype(np.float32)
        print("class weights:", np.array2string(self.class_weights, precision=3))
                
    def log_model(self):
        with self.log_file.open('w', encoding='utf-8') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            fh.write("\narchitecture: " + str(self.architecture))
            fh.write("\nrnn_type: " + str(self.rnn_type))
            fh.write("\nrnn_units: " + str(self.rnn_units))
            fh.write("\nrnn_layers: " + str(self.rnn_layers))
            fh.write("\nrnn_recurrent_dropout: " + str(self.rnn_recurrent_dropout))
            fh.write("\nbidirectional: " + str(self.bidirectional) + "\n")
            fh.write("\nuse_early_stopping: " + str(self.use_early_stopping))
            fh.write("\nearly_stopping_patience: " + str(self.early_stopping_patience))
            fh.write("\nheldout_val_fraction: " + str(self.heldout_val_fraction))
            fh.write("\nheldout_split_seed: " + str(self.heldout_split_seed))
            if hasattr(self, "partition"):
                fh.write("\ntrain_samples: " + str(len(self.partition.get("training", []))))
                fh.write("\nval_samples: " + str(len(self.partition.get("validation", []))))
                fh.write("\ntest_samples: " + str(len(self.partition.get("test", []))))
            fh.write("\ngenerator_cache_files: " + str(self.generator_cache_files))
            fh.write("\nuse_tf_data: " + str(self.use_tf_data))
            fh.write("\nenable_mixed_precision: " + str(self.enable_mixed_precision))
            fh.write("\nenable_xla: " + str(self.enable_xla))
            fh.write("\ngpu_memory_growth: " + str(self.gpu_memory_growth))
            fh.write("\nuse_class_weighting: " + str(self.use_class_weighting))
            fh.write("\nclass_weight_power: " + str(self.class_weight_power))
            fh.write("\nmin_class_weight: " + str(self.min_class_weight))
            fh.write("\nmax_class_weight: " + str(self.max_class_weight))
            fh.write("\nclosed_class_weight: " + str(self.closed_class_weight))
            fh.write("\nuse_sounding_aux_loss: " + str(self.use_sounding_aux_loss))
            fh.write("\nsounding_aux_weight: " + str(self.sounding_aux_weight))
            fh.write("\nsounding_positive_weight: " + str(self.sounding_positive_weight))
            fh.write("\noptimizer_learning_rate: " + str(self.optimizer_learning_rate))
            fh.write("\ntrack_validation_diagnostics: " + str(self.track_validation_diagnostics))
            fh.write("\ncheckpoint_validation_metrics: " + str(self.checkpoint_validation_metrics))
            fh.write("\nclass_weights: " + np.array2string(self.class_weights, precision=3))
            fh.write("\nbackend_is_tensorflow: " + str(self._tf_backend))
            fh.write("\ngpu_count: " + str(len(self._gpu_devices)) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
       
    def softmax_by_string(self, t):
        return keras.ops.softmax(t, axis=-1)
    
    def catcross_by_string(self, target, output):
        per_string_loss = keras.losses.categorical_crossentropy(target, output)
        if self.use_class_weighting or self.closed_class_weight is not None:
            class_weights = keras.ops.cast(self.class_weights, dtype=per_string_loss.dtype)
            sample_weights = keras.ops.sum(target * class_weights, axis=-1)
            per_string_loss = per_string_loss * sample_weights
        total_loss = keras.ops.sum(per_string_loss, axis=1)

        if self.use_sounding_aux_loss:
            eps = keras.ops.cast(1e-7, dtype=output.dtype)
            # y=1 means string is sounding (open/fretted), y=0 means closed/muted.
            target_sounding = keras.ops.sum(target[:, :, 1:], axis=-1)
            pred_sounding = keras.ops.clip(1.0 - output[:, :, 0], eps, 1.0 - eps)
            pos_w = keras.ops.cast(self.sounding_positive_weight, dtype=pred_sounding.dtype)
            neg_w = keras.ops.cast(1.0, dtype=pred_sounding.dtype)
            sounding_loss = -(
                (pos_w * target_sounding * keras.ops.log(pred_sounding)) +
                (neg_w * (1.0 - target_sounding) * keras.ops.log(1.0 - pred_sounding))
            )
            total_loss = total_loss + (self.sounding_aux_weight * keras.ops.sum(sounding_loss, axis=1))

        return total_loss
    
    def avg_acc(self, y_true, y_pred):
        per_string_match = keras.ops.equal(
            keras.ops.argmax(y_true, axis=-1),
            keras.ops.argmax(y_pred, axis=-1)
        )
        return keras.ops.mean(keras.ops.cast(per_string_match, "float32"))
           
    def build_cnn_model(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes * self.num_strings)(x)
        x = Reshape((self.num_strings, self.num_classes))(x)
        outputs = Activation(self.softmax_by_string)(x)
        return keras.Model(inputs=inputs, outputs=outputs, name="tabcnn")

    def _build_recurrent_layer(self, return_sequences):
        if self.rnn_type == "lstm":
            return LSTM(
                self.rnn_units,
                return_sequences=return_sequences,
                recurrent_dropout=self.rnn_recurrent_dropout,
            )
        return GRU(
            self.rnn_units,
            return_sequences=return_sequences,
            recurrent_dropout=self.rnn_recurrent_dropout,
        )

    def build_crnn_model(self):
        inputs = Input(shape=self.input_shape)

        # Reduce only the frequency axis so the RNN can read a temporal sequence.
        x = Conv2D(32, kernel_size=(5, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 1))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 1))(x)
        x = Dropout(0.25)(x)

        # Convert (freq, time, channels) -> (time, flattened_features).
        x = Permute((2, 1, 3))(x)
        x = TimeDistributed(Flatten())(x)

        for layer_idx in range(self.rnn_layers):
            return_sequences = layer_idx < (self.rnn_layers - 1)
            rnn_layer = self._build_recurrent_layer(return_sequences=return_sequences)
            if self.bidirectional:
                x = Bidirectional(rnn_layer)(x)
            else:
                x = rnn_layer(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes * self.num_strings)(x)
        x = Reshape((self.num_strings, self.num_classes))(x)
        outputs = Activation(self.softmax_by_string)(x)
        return keras.Model(inputs=inputs, outputs=outputs, name="tabcrnn")

    def build_model(self):
        keras.backend.clear_session()
        if self.architecture == "cnn":
            model = self.build_cnn_model()
        else:
            model = self.build_crnn_model()

        compile_kwargs = {
            "loss": self.catcross_by_string,
            "optimizer": keras.optimizers.Adadelta(learning_rate=self.optimizer_learning_rate),
            "metrics": [self.avg_acc],
        }
        if self._tf_backend:
            compile_kwargs["jit_compile"] = bool(self.enable_xla)
        model.compile(**compile_kwargs)
        
        self.model = model

    def _sequence_to_tf_dataset(self, sequence):
        if not (self.use_tf_data and self._tf_backend and tf is not None):
            return sequence

        output_signature = (
            tf.TensorSpec(shape=(None,) + self.input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.num_strings, self.num_classes), dtype=tf.float32),
        )

        def _batch_iter():
            for batch_idx in range(len(sequence)):
                yield sequence[batch_idx]

        dataset = tf.data.Dataset.from_generator(
            _batch_iter,
            output_signature=output_signature,
        )
        # Keras expects at least `steps_per_epoch * epochs` batches when
        # `steps_per_epoch` is provided. Repeat the finite sequence-backed
        # dataset and let `steps_per_epoch` delimit each epoch.
        return dataset.repeat().prefetch(tf.data.AUTOTUNE)

    def _sequence_epoch_callback(self, sequence):
        if not (self.use_tf_data and self._tf_backend and tf is not None):
            return None

        class _SequenceEpochEndCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                sequence.on_epoch_end()

        return _SequenceEpochEndCallback()

    def _predict_generator(self, generator):
        x_batches = []
        y_batches = []
        for batch_idx in range(len(generator)):
            batch_x, batch_y = generator[batch_idx]
            x_batches.append(batch_x)
            y_batches.append(batch_y)

        if not x_batches:
            raise ValueError("Generator produced no batches for prediction.")

        x_all = np.concatenate(x_batches, axis=0)
        y_all = np.concatenate(y_batches, axis=0)
        y_pred = self.model.predict(x_all, verbose=0)
        return y_pred, y_all

    def _compute_eval_metrics(self, y_pred, y_true):
        return {
            "pp": pitch_precision(y_pred, y_true),
            "pr": pitch_recall(y_pred, y_true),
            "pf": pitch_f_measure(y_pred, y_true),
            "tp": tab_precision(y_pred, y_true),
            "tr": tab_recall(y_pred, y_true),
            "tf": tab_f_measure(y_pred, y_true),
            "tdr": tab_disamb(y_pred, y_true),
            "ind": incorrect_note_distance(y_pred, y_true),
        }

    def _write_confusion_reports(self, y_pred, y_true, epoch):
        confusion = fret_confusion_matrix(y_pred, y_true, num_classes=self.num_classes)
        matrix_df = pd.DataFrame(
            confusion,
            index=[f"true_{idx}" for idx in range(self.num_classes)],
            columns=[f"pred_{idx}" for idx in range(self.num_classes)],
        )
        matrix_df.to_csv(self.confusion_csv_dir / f"val_fret_confusion_epoch_{epoch + 1:02d}.csv")
        self._save_confusion_heatmap(confusion, epoch)

        summary_rows = fret_confusion_summary(y_pred, y_true, num_classes=self.num_classes)
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                by=["is_correct", "count", "true_fret", "pred_fret"],
                ascending=[True, False, True, True],
            )
        summary_df.to_csv(
            self.confusion_csv_dir / f"val_fret_confusion_summary_epoch_{epoch + 1:02d}.csv",
            index=False,
        )

    def _save_confusion_heatmap(self, confusion, epoch):
        save_confusion_heatmap(
            confusion,
            self.confusion_plot_dir / f"val_fret_confusion_epoch_{epoch + 1:02d}.png",
            f"{self.architecture.upper()} fold {self.data_split}: validation fret confusion epoch {epoch + 1}",
            self.num_classes,
        )

    def _validation_diagnostics_callback(self):
        if not self.track_validation_diagnostics:
            return None

        outer = self

        class _ValidationDiagnosticsCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best_metrics = {"pf": float("-inf"), "tf": float("-inf")}
                self.metrics_path = outer.metrics_dir / "validation_metrics.csv"
                self.header_written = False

            def on_train_begin(self, logs=None):
                self.header_written = self.metrics_path.exists() and self.metrics_path.stat().st_size > 0

            def on_epoch_end(self, epoch, logs=None):
                logs = logs if logs is not None else {}
                y_pred, y_true = outer._predict_generator(outer.validation_eval_generator)
                metrics = outer._compute_eval_metrics(y_pred, y_true)
                for metric_name, metric_value in metrics.items():
                    logs[f"val_{metric_name}"] = metric_value

                outer._write_confusion_reports(y_pred, y_true, epoch)
                self._append_metrics_row(epoch, metrics)

                if outer.checkpoint_validation_metrics:
                    self._maybe_save_checkpoint(epoch, "pf", metrics["pf"])
                    self._maybe_save_checkpoint(epoch, "tf", metrics["tf"])

            def _append_metrics_row(self, epoch, metrics):
                row = {"epoch": epoch + 1}
                row.update({f"val_{name}": value for name, value in metrics.items()})
                fieldnames = ["epoch"] + [f"val_{name}" for name in metrics.keys()]
                with outer.metrics_path_for_append(self.metrics_path).open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    if not self.header_written:
                        writer.writeheader()
                        self.header_written = True
                    writer.writerow(row)

            def _maybe_save_checkpoint(self, epoch, metric_name, metric_value):
                if metric_value <= self.best_metrics[metric_name]:
                    return
                self.best_metrics[metric_name] = metric_value
                checkpoint_path = outer.checkpoints_dir / f"best_val_{metric_name}.weights.h5"
                outer.model.save_weights(str(checkpoint_path))
                marker_path = outer.checkpoints_dir / f"best_val_{metric_name}.txt"
                marker_path.write_text(
                    f"epoch,{epoch + 1}\nval_{metric_name},{metric_value:.10f}\n",
                    encoding="utf-8",
                )

        return _ValidationDiagnosticsCallback()

    def metrics_path_for_append(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def train(self):
        train_data = self._sequence_to_tf_dataset(self.training_generator)
        validation_data = self.validation_fit_generator
        callbacks = []
        validation_diagnostics = self._validation_diagnostics_callback()
        if validation_diagnostics is not None:
            callbacks.append(validation_diagnostics)
        callbacks.append(keras.callbacks.CSVLogger(str(self.metrics_dir / "history.csv")))
        if self.use_early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                )
            )
        fit_kwargs = {
            "validation_data": validation_data,
            "epochs": self.epochs,
            "verbose": 1,
            "callbacks": callbacks,
        }
        if train_data is not self.training_generator:
            fit_kwargs["steps_per_epoch"] = len(self.training_generator)
        fit_kwargs["validation_steps"] = len(self.validation_fit_generator)

        if train_data is not self.training_generator:
            epoch_callback = self._sequence_epoch_callback(self.training_generator)
            if epoch_callback is not None:
                callbacks.append(epoch_callback)

        history = self.model.fit(train_data, **fit_kwargs)
        self.save_history_plots(history.history)

    def save_history_plots(self, history):
        loss = history.get("loss")
        val_loss = history.get("val_loss")
        if loss is not None or val_loss is not None:
            reference = loss if loss is not None else val_loss
            epochs = np.arange(1, len(reference) + 1)

            fig, ax_loss = plt.subplots(figsize=(8, 5))
            if loss is not None:
                ax_loss.plot(epochs, loss, label="loss", linewidth=2, color="tab:blue")
            if val_loss is not None:
                ax_loss.plot(epochs, val_loss, label="val_loss", linewidth=2, color="tab:orange")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(alpha=0.3)

            ax_metric = ax_loss.twinx()
            val_pf = history.get("val_pf")
            val_tf = history.get("val_tf")
            if val_pf is not None:
                ax_metric.plot(epochs, val_pf, label="val_pf", linewidth=2, color="tab:green")
            if val_tf is not None:
                ax_metric.plot(epochs, val_tf, label="val_tf", linewidth=2, color="tab:red")
            ax_metric.set_ylabel("Validation F-measure")

            handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
            handles_metric, labels_metric = ax_metric.get_legend_handles_labels()
            ax_loss.legend(handles_loss + handles_metric, labels_loss + labels_metric, loc="best")
            plt.title(f"{self.architecture.upper()} fold {self.data_split}: loss and validation F-measures")
            fig.tight_layout()
            fig.savefig(self.plots_dir / "loss_curve.png", dpi=160)
            plt.close(fig)

        accuracy = history.get("avg_acc")
        val_accuracy = history.get("val_avg_acc")
        if accuracy is not None or val_accuracy is not None:
            reference = accuracy if accuracy is not None else val_accuracy
            epochs = np.arange(1, len(reference) + 1)

            plt.figure(figsize=(8, 5))
            if accuracy is not None:
                plt.plot(epochs, accuracy, label="avg_acc", linewidth=2)
            if val_accuracy is not None:
                plt.plot(epochs, val_accuracy, label="val_avg_acc", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("accuracy")
            plt.title(f"{self.architecture.upper()} fold {self.data_split}: avg_acc")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / "avg_acc_curve.png", dpi=160)
            plt.close()
        
    def save_weights(self):
        self.model.save_weights(str(self.artifacts_dir / "weights.weights.h5"))

    def preflight_save_check(self):
        probe_path = self.artifacts_dir / "_save_probe.weights.h5"
        self.model.save_weights(str(probe_path))
        probe_path.unlink(missing_ok=True)
        
    def test(self):
        self.X_test, self.y_gt = self.test_generator[0]
        self.y_pred = self.model.predict(self.X_test)
        
    def save_predictions(self):
        np.savez(self.artifacts_dir / "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)
        
    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))
        self.metrics["ind"].append(incorrect_note_distance(self.y_pred, self.y_gt))
        
    def save_results_csv(self):
        output = {}
        row_labels = None
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
                if row_labels is None:
                    row_labels = [f"g{i}" for i in range(len(vals))] + ["mean", "std dev"]
        output["data"] = row_labels if row_labels is not None else ["mean", "std dev"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder / "results.csv") 


def evaluate_saved_run(run_path):
    run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    fold_dirs = sorted(
        [p for p in run_path.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not fold_dirs:
        raise FileNotFoundError(
            f"No numeric fold directories found in {run_path} (expected e.g. 0, 1, ...)."
        )

    metric_fns = {
        "pp": pitch_precision,
        "pr": pitch_recall,
        "pf": pitch_f_measure,
        "tp": tab_precision,
        "tr": tab_recall,
        "tf": tab_f_measure,
        "tdr": tab_disamb,
        "ind": incorrect_note_distance,
    }

    rows = []
    row_labels = []
    for fold_dir in fold_dirs:
        pred_file = fold_dir / "artifacts" / "predictions.npz"
        if not pred_file.exists():
            pred_file = fold_dir / "predictions.npz"
        if not pred_file.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_file}")
        with np.load(pred_file, allow_pickle=False) as loaded:
            y_pred = loaded["y_pred"]
            y_gt = loaded["y_gt"]
        rows.append({name: fn(y_pred, y_gt) for name, fn in metric_fns.items()})
        row_labels.append(f"g{fold_dir.name}")

    df = pd.DataFrame(rows, index=row_labels)
    mean_row = df.mean(axis=0)
    std_row = df.std(axis=0, ddof=0)
    out_df = pd.concat(
        [
            df.reset_index(names="data"),
            pd.DataFrame([{"data": "mean", **mean_row.to_dict()}]),
            pd.DataFrame([{"data": "std dev", **std_row.to_dict()}]),
        ],
        ignore_index=True,
    )
    out_df = out_df[["pp", "pr", "pf", "tp", "tr", "tf", "tdr", "ind", "data"]]
    out_df.to_csv(run_path / "results.csv")

    print("\nPer-fold metrics")
    print(df.to_string())
    print("\nMean")
    print(mean_row.to_string())
    print("\nStd dev")
    print(std_row.to_string())
    print(f"\nSaved results to {run_path / 'results.csv'}")
    return out_df


def render_saved_confusions(run_path):
    run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    fold_dirs = sorted(
        [p for p in run_path.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not fold_dirs:
        raise FileNotFoundError(
            f"No numeric fold directories found in {run_path} (expected e.g. 0, 1, ...)."
        )

    rendered = []
    for fold_dir in fold_dirs:
        csv_dirs = [fold_dir / "confusion" / "csv", fold_dir]
        plot_dir = fold_dir / "confusion" / "png"
        for csv_dir in csv_dirs:
            if not csv_dir.exists():
                continue
            for csv_path in sorted(csv_dir.glob("val_fret_confusion_epoch_*.csv")):
                frame = pd.read_csv(csv_path, index_col=0)
                confusion = frame.to_numpy()
                output_path = plot_dir / csv_path.with_suffix(".png").name
                epoch_label = csv_path.stem.split("_")[-1]
                save_confusion_heatmap(
                    confusion,
                    output_path,
                    f"Saved fold {fold_dir.name}: validation fret confusion epoch {int(epoch_label)}",
                    confusion.shape[0],
                )
                rendered.append(output_path)

    if not rendered:
        raise FileNotFoundError(f"No saved confusion CSVs found under {run_path}.")

    print("Rendered confusion heatmaps:")
    for path in rendered:
        print(path)
    return rendered


def parse_args():
    parser = argparse.ArgumentParser(description="Train TabCNN or evaluate saved predictions.")
    parser.add_argument(
        "--evaluate-only",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Path to a completed run directory (contains fold subdirs with predictions.npz).",
    )
    parser.add_argument(
        "--render-confusions-only",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Path to a saved run directory containing confusion CSVs to render into PNGs.",
    )
    return parser.parse_args()
        
##################################
########### EXPERIMENT ###########
##################################

if __name__ == "__main__":
    args = parse_args()
    if args.evaluate_only:
        evaluate_saved_run(args.evaluate_only)
        raise SystemExit(0)
    if args.render_confusions_only:
        render_saved_confusions(args.render_confusions_only)
        raise SystemExit(0)

    tabcnn = TabCNN(
        architecture="crnn",
        rnn_type="gru",
        rnn_units=64,
        rnn_layers=1,
        bidirectional=False,
        rnn_recurrent_dropout=0.2,
        use_early_stopping=False,
        early_stopping_patience=2,
        enable_mixed_precision=False,
        use_class_weighting=False,
        use_sounding_aux_loss=True,
        sounding_aux_weight=0.5,
        sounding_positive_weight=2.0,
        optimizer_learning_rate=1.0,
    )

    print("logging model...")
    tabcnn.build_model()
    tabcnn.log_model()

    for fold in range(6):
        print("\nfold " + str(fold))
        tabcnn.partition_data(fold)
        print("building model...")
        tabcnn.build_model()
        print("checking save path...")
        tabcnn.preflight_save_check()
        print("training...")
        tabcnn.train()
        tabcnn.save_weights()
        print("testing...")
        tabcnn.test()
        tabcnn.save_predictions()
        print("evaluation...")
        tabcnn.evaluate()
    print("saving results...")
    tabcnn.save_results_csv()
