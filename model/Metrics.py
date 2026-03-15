import numpy as np
import keras


STRING_MIDI_PITCHES = np.array([40, 45, 50, 55, 59, 64], dtype=np.int32)
    
def tab2pitch(tab):
    pitch_vector = np.zeros(44)
    string_pitches = [40, 45, 50, 55, 59, 64]
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = np.argmax(fret_vector, -1)
        # 0 means that the string is closed 
        if fret_class > 0:
            pitch_num = fret_class + string_pitches[string_num] - 41
            pitch_vector[pitch_num] = 1
    return pitch_vector

def tab2bin(tab):
    tab_arr = np.zeros((6,20))
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = np.argmax(fret_vector, -1)
        # 0 means that the string is closed 
        if fret_class > 0:
            fret_num = fret_class - 1
            tab_arr[string_num][fret_num] = 1
    return tab_arr

def pitch_precision(pred, gt):
    pitch_pred = np.array([tab2pitch(tab) for tab in pred])
    pitch_gt = np.array([tab2pitch(tab) for tab in gt])
    numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
    denominator = np.sum(pitch_pred.flatten())
    return (1.0 * numerator) / denominator

def pitch_recall(pred, gt):
    pitch_pred = np.array([tab2pitch(tab) for tab in pred])
    pitch_gt = np.array([tab2pitch(tab) for tab in gt])
    numerator = np.sum(np.multiply(pitch_pred, pitch_gt).flatten())
    denominator = np.sum(pitch_gt.flatten())
    return (1.0 * numerator) / denominator

def pitch_f_measure(pred, gt):
    p = pitch_precision(pred, gt)
    r = pitch_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f

def tab_precision(pred, gt):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = np.array([tab2bin(tab) for tab in pred])
    tab_gt = np.array([tab2bin(tab) for tab in gt])
    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_pred.flatten())
    return (1.0 * numerator) / denominator

def tab_recall(pred, gt):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = np.array([tab2bin(tab) for tab in pred])
    tab_gt = np.array([tab2bin(tab) for tab in gt])
    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_gt.flatten())
    return (1.0 * numerator) / denominator

def tab_f_measure(pred, gt):
    p = tab_precision(pred, gt)
    r = tab_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f

def tab_disamb(pred, gt):
    tp = tab_precision(pred, gt)
    pp = pitch_precision(pred, gt)
    return tp / pp


def incorrect_note_distance(pred, gt):
    """Average semitone distance on wrong per-string note predictions.

    This metric only considers positions where:
    - the predicted class and ground-truth class differ, and
    - both classes represent sounding notes (class > 0, i.e. not "closed").

    It measures how far the wrong guess is from the correct note in semitones.
    Lower is better. Returns 0.0 if there are no qualifying mistakes.
    """
    pred_cls = np.argmax(pred, axis=-1)
    gt_cls = np.argmax(gt, axis=-1)

    wrong_mask = pred_cls != gt_cls
    note_mask = (pred_cls > 0) & (gt_cls > 0)
    valid_mask = wrong_mask & note_mask

    if not np.any(valid_mask):
        return 0.0

    # Class 1 is open string, so MIDI pitch = open-string MIDI + (class - 1).
    pred_midi = STRING_MIDI_PITCHES[None, :] + (pred_cls - 1)
    gt_midi = STRING_MIDI_PITCHES[None, :] + (gt_cls - 1)
    distances = np.abs(pred_midi - gt_midi)
    return float(np.mean(distances[valid_mask]))


def fret_confusion_matrix(pred, gt, num_classes=21):
    """Count true-vs-predicted fret classes across all frames and strings."""
    pred_cls = np.argmax(pred, axis=-1).reshape(-1)
    gt_cls = np.argmax(gt, axis=-1).reshape(-1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (gt_cls, pred_cls), 1)
    return confusion


def fret_confusion_summary(pred, gt, num_classes=21):
    """Return a dataframe-friendly summary of non-zero fret confusions."""
    confusion = fret_confusion_matrix(pred, gt, num_classes=num_classes)
    rows = []
    for true_fret in range(num_classes):
        for pred_fret in range(num_classes):
            count = int(confusion[true_fret, pred_fret])
            if count == 0:
                continue
            rows.append(
                {
                    "true_fret": true_fret,
                    "pred_fret": pred_fret,
                    "count": count,
                    "is_correct": true_fret == pred_fret,
                }
            )
    return rows
