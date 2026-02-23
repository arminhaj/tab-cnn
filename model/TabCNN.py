''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
'''

from __future__ import print_function
import keras
from pathlib import Path
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - tensorflow may be absent in some environments
    tf = None
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Permute, TimeDistributed
from keras.layers import GRU, LSTM, Bidirectional
from DataGenerator import DataGenerator
import pandas as pd
import numpy as np
import datetime
from Metrics import *

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
                 generator_cache_files=32,
                 use_tf_data=True,
                 enable_mixed_precision=True,
                 enable_xla=False,
                 gpu_memory_growth=True):   
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.architecture = architecture.lower()
        self.rnn_type = rnn_type.lower()
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.generator_cache_files = generator_cache_files
        self.use_tf_data = use_tf_data
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_xla = enable_xla
        self.gpu_memory_growth = gpu_memory_growth
        self._tf_backend = False
        self._gpu_devices = []

        if self.architecture not in {"cnn", "crnn"}:
            raise ValueError("architecture must be 'cnn' or 'crnn'")
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        if self.architecture == "crnn" and self.rnn_layers < 1:
            raise ValueError("rnn_layers must be >= 1 when using architecture='crnn'")

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
        self.partition["validation"] = []
        for ID in self.list_IDs:
            guitarist = int(ID.split("_")[0])
            if guitarist == data_split:
                self.partition["validation"].append(ID)
            else:
                self.partition["training"].append(ID)
                
        self.training_generator = DataGenerator(self.partition['training'], 
                                                data_path=self.generator_data_path(), 
                                                batch_size=self.batch_size, 
                                                shuffle=True,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size,
                                                cache_files=self.generator_cache_files)
        
        self.validation_generator = DataGenerator(self.partition['validation'], 
                                                data_path=self.generator_data_path(), 
                                                batch_size=len(self.partition['validation']), 
                                                shuffle=False,
                                                spec_repr=self.spec_repr, 
                                                con_win_size=self.con_win_size,
                                                cache_files=self.generator_cache_files)
        
        self.split_folder = self.save_folder / str(self.data_split)
        self.split_folder.mkdir(parents=True, exist_ok=True)
                
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
            fh.write("\nbidirectional: " + str(self.bidirectional) + "\n")
            fh.write("\ngenerator_cache_files: " + str(self.generator_cache_files))
            fh.write("\nuse_tf_data: " + str(self.use_tf_data))
            fh.write("\nenable_mixed_precision: " + str(self.enable_mixed_precision))
            fh.write("\nenable_xla: " + str(self.enable_xla))
            fh.write("\ngpu_memory_growth: " + str(self.gpu_memory_growth))
            fh.write("\nbackend_is_tensorflow: " + str(self._tf_backend))
            fh.write("\ngpu_count: " + str(len(self._gpu_devices)) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
       
    def softmax_by_string(self, t):
        return keras.ops.softmax(t, axis=-1)
    
    def catcross_by_string(self, target, output):
        per_string_loss = keras.losses.categorical_crossentropy(target, output)
        return keras.ops.sum(per_string_loss, axis=1)
    
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
            return LSTM(self.rnn_units, return_sequences=return_sequences)
        return GRU(self.rnn_units, return_sequences=return_sequences)

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
        if self.architecture == "cnn":
            model = self.build_cnn_model()
        else:
            model = self.build_crnn_model()

        compile_kwargs = {
            "loss": self.catcross_by_string,
            "optimizer": keras.optimizers.Adadelta(),
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

    def train(self):
        train_data = self._sequence_to_tf_dataset(self.training_generator)
        fit_kwargs = {
            "validation_data": None,
            "epochs": self.epochs,
            "verbose": 1,
        }
        if train_data is not self.training_generator:
            fit_kwargs["steps_per_epoch"] = len(self.training_generator)
            epoch_callback = self._sequence_epoch_callback(self.training_generator)
            if epoch_callback is not None:
                fit_kwargs["callbacks"] = [epoch_callback]
        self.model.fit(train_data, **fit_kwargs)
        
    def save_weights(self):
        self.model.save_weights(str(self.split_folder / "weights.weights.h5"))

    def preflight_save_check(self):
        probe_path = self.split_folder / "_save_probe.weights.h5"
        self.model.save_weights(str(probe_path))
        probe_path.unlink(missing_ok=True)
        
    def test(self):
        self.X_test, self.y_gt = self.validation_generator[0]
        self.y_pred = self.model.predict(self.X_test)
        
    def save_predictions(self):
        np.savez(self.split_folder / "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)
        
    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))
        
    def save_results_csv(self):
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        output["data"] =  self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder / "results.csv") 
        
##################################
########### EXPERIMENT ###########
##################################

if __name__ == "__main__":
    tabcnn = TabCNN(architecture="crnn", rnn_type="gru", rnn_units=128, rnn_layers=1, bidirectional=True)

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
