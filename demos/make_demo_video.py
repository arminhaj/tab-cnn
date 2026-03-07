"""Render tablature demo visualizations from saved fold predictions.

Styles:
- packaged: single fretboard + right-side legend (matches original demo style)
- split: side-by-side ground-truth vs prediction grids

It always writes an animated GIF preview. If `ffmpeg` is available, it can
also render an MP4 and optionally mux audio from GuitarSet.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import jams
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None


SR_DOWNSAMPLED = 22050
HOP_LENGTH = 512
FRAME_RATE = SR_DOWNSAMPLED / float(HOP_LENGTH)
NUM_STRINGS = 6
NUM_CLASSES = 21
NUM_FRETS = 20
STRING_MIDI_PITCHES = [40, 45, 50, 55, 59, 64]
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]


def parse_id_row(id_row: str) -> tuple[str, int]:
    track, frame = id_row.rsplit("_", 1)
    return track, int(frame)


def load_validation_ids(id_csv: Path, fold: int) -> list[str]:
    ids = [line.strip() for line in id_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for row in ids:
        guitarist = int(row.split("_", 1)[0])
        if guitarist == fold:
            out.append(row)
    return out


def list_tracks_in_ids(validation_ids: list[str]) -> list[str]:
    tracks = []
    seen = set()
    for row in validation_ids:
        track, _ = parse_id_row(row)
        if track not in seen:
            seen.add(track)
            tracks.append(track)
    return tracks


def load_track_slice(
    validation_ids: list[str], track: str, y_pred: np.ndarray, y_gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = []
    frame_numbers = []
    for idx, row in enumerate(validation_ids):
        row_track, frame_num = parse_id_row(row)
        if row_track == track:
            indices.append(idx)
            frame_numbers.append(frame_num)

    if not indices:
        raise ValueError(f"Track '{track}' not found in this fold's validation IDs.")

    order = np.argsort(frame_numbers)
    selected = np.array(indices, dtype=np.int64)[order]
    selected_frame_numbers = np.array(frame_numbers, dtype=np.int64)[order]

    return y_pred[selected], y_gt[selected], selected_frame_numbers


def midi_pitch_to_class_idx(midi_pitch: float | None, string_num: int) -> int:
    if midi_pitch is None:
        return 0
    fret = int(round(float(midi_pitch)) - STRING_MIDI_PITCHES[string_num])
    if fret < 0 or fret >= NUM_FRETS:
        return 0
    return fret + 1


def load_gt_from_jams(annotation_root: Path, track: str, frame_numbers: np.ndarray) -> np.ndarray:
    jam_path = annotation_root / f"{track}.jams"
    if not jam_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {jam_path}")

    jam = jams.load(str(jam_path))
    note_midi_annos = jam.annotations["note_midi"]
    if len(note_midi_annos) < NUM_STRINGS:
        raise ValueError(
            f"Expected at least {NUM_STRINGS} note_midi annotations in {jam_path}, found {len(note_midi_annos)}."
        )

    times = frame_numbers.astype(np.float64) * (HOP_LENGTH / float(SR_DOWNSAMPLED))
    gt_classes = np.zeros((len(frame_numbers), NUM_STRINGS), dtype=np.int64)
    for string_num in range(NUM_STRINGS):
        sampled = note_midi_annos[string_num].to_samples(times)
        for idx, sample in enumerate(sampled):
            midi = sample[0] if sample else None
            gt_classes[idx, string_num] = midi_pitch_to_class_idx(midi, string_num)
    return gt_classes


def class_to_tab_matrix(class_idx_per_string: np.ndarray) -> np.ndarray:
    tab = np.zeros((NUM_STRINGS, NUM_FRETS), dtype=np.uint8)
    for s in range(NUM_STRINGS):
        cls = int(class_idx_per_string[s])
        if cls > 0:
            fret = cls - 1
            if 0 <= fret < NUM_FRETS:
                tab[s, fret] = 1
    return tab


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSerif.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def prepare_frames_dir(out_frames_dir: Path) -> None:
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_file in out_frames_dir.glob("frame_*.png"):
        frame_file.unlink()


def draw_single_tab_split(draw: ImageDraw.ImageDraw, x0: int, y0: int, title: str, cls: np.ndarray) -> None:
    cell_w = 22
    cell_h = 24
    top_pad = 36
    draw.text((x0, y0), title, fill=(20, 20, 20))
    grid_y = y0 + top_pad
    tab = class_to_tab_matrix(cls)

    for s in range(NUM_STRINGS):
        for f in range(NUM_FRETS):
            x1 = x0 + f * cell_w
            y1 = grid_y + s * cell_h
            x2 = x1 + cell_w - 2
            y2 = y1 + cell_h - 2

            if tab[s, f]:
                fill = (66, 133, 244) if title == "Prediction" else (46, 125, 50)
            else:
                fill = (247, 247, 247)

            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=(210, 210, 210), width=1)

    for s in range(NUM_STRINGS):
        draw.text((x0 - 20, grid_y + s * cell_h + 5), str(6 - s), fill=(80, 80, 80))

    for f in range(0, NUM_FRETS, 2):
        draw.text((x0 + f * cell_w + 4, grid_y + NUM_STRINGS * cell_h + 4), str(f), fill=(100, 100, 100))


def get_note_y_for_fret(fret: int, y_top: int, fret_h: int) -> int:
    if fret == 0:
        return y_top - int(0.45 * fret_h)
    return int(y_top + (fret - 0.5) * fret_h)


def draw_x_marker(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color: tuple[int, int, int], width: int) -> None:
    draw.line([(x - size, y - size), (x + size, y + size)], fill=color, width=width)
    draw.line([(x - size, y + size), (x + size, y - size)], fill=color, width=width)


def draw_packaged_frame(
    draw: ImageDraw.ImageDraw,
    pred_cls: np.ndarray,
    gt_cls: np.ndarray,
    frame_idx: int,
    display_frets: int,
) -> None:
    bg = (233, 233, 233)
    black = (20, 20, 20)
    blue = (25, 45, 230)
    magenta = (220, 40, 230)
    red = (235, 20, 20)
    gray = (145, 145, 145)

    width, height = 1080, 1080
    draw.rectangle([0, 0, width, height], fill=bg)
    draw.line([(540, 0), (540, height)], fill=(110, 110, 110), width=2)

    x_left, x_right = 92, 408
    y_top = 102
    fret_h = 84
    y_bottom = y_top + display_frets * fret_h
    string_gap = (x_right - x_left) / float(NUM_STRINGS - 1)
    string_x = [int(round(x_left + i * string_gap)) for i in range(NUM_STRINGS)]

    draw.line([(x_left - 14, y_top), (x_right + 14, y_top)], fill=black, width=6)
    for fret in range(1, display_frets + 1):
        y = y_top + fret * fret_h
        draw.line([(x_left - 14, y), (x_right + 14, y)], fill=black, width=3)
    for x in string_x:
        draw.line([(x, y_top - 8), (x, y_bottom)], fill=gray, width=3)

    fret_label_font = load_font(44)
    string_label_font = load_font(44)
    for fret in range(1, display_frets + 1):
        y = int(y_top + (fret - 0.5) * fret_h - 20)
        draw.text((48, y), str(fret), fill=(35, 35, 35), font=fret_label_font)
    for i, name in enumerate(STRING_NAMES):
        draw.text((string_x[i] - 12, y_bottom + 20), name, fill=(35, 35, 35), font=string_label_font)

    note_r = 13
    ring_w = 5
    x_size = 10
    x_y = y_top - 44

    for s in range(NUM_STRINGS):
        gt_val = int(gt_cls[s])
        pred_val = int(pred_cls[s])

        if gt_val == 0:
            draw_x_marker(draw, string_x[s], x_y, x_size, blue, 3)
        else:
            gt_fret = min(max(gt_val - 1, 0), display_frets)
            gt_y = get_note_y_for_fret(gt_fret, y_top, fret_h)
            draw.ellipse(
                [string_x[s] - note_r, gt_y - note_r, string_x[s] + note_r, gt_y + note_r],
                fill=blue,
            )

        if pred_val == gt_val:
            if pred_val == 0:
                draw_x_marker(draw, string_x[s], x_y, x_size + 2, magenta, 2)
                draw_x_marker(draw, string_x[s], x_y, x_size, blue, 3)
            else:
                pred_fret = min(max(pred_val - 1, 0), display_frets)
                pred_y = get_note_y_for_fret(pred_fret, y_top, fret_h)
                draw.ellipse(
                    [string_x[s] - note_r - ring_w, pred_y - note_r - ring_w, string_x[s] + note_r + ring_w, pred_y + note_r + ring_w],
                    outline=magenta,
                    width=ring_w,
                )
                draw.ellipse(
                    [string_x[s] - note_r, pred_y - note_r, string_x[s] + note_r, pred_y + note_r],
                    fill=blue,
                )
        elif pred_val == 0:
            draw_x_marker(draw, string_x[s], x_y, x_size + 2, red, 3)
        else:
            pred_fret = min(max(pred_val - 1, 0), display_frets)
            pred_y = get_note_y_for_fret(pred_fret, y_top, fret_h)
            draw.ellipse(
                [string_x[s] - note_r, pred_y - note_r, string_x[s] + note_r, pred_y + note_r],
                fill=red,
            )

    title_font = load_font(78)
    body_font = load_font(56)
    draw.text((690, 42), "Legend", fill=black, font=title_font)
    draw.line([(690, 122), (918, 122)], fill=black, width=4)

    y0 = 260
    draw.ellipse([592 - 20, y0 - 20, 592 + 20, y0 + 20], fill=blue)
    draw.text((650, y0 - 34), "Ground truth labels are", fill=black, font=body_font)
    draw.text((650, y0 + 30), "shown in", fill=black, font=body_font)
    draw.text((848, y0 + 30), "blue.", fill=blue, font=body_font)

    y1 = 470
    draw.ellipse([592 - 20, y1 - 20, 592 + 20, y1 + 20], fill=blue)
    draw.ellipse([592 - 28, y1 - 28, 592 + 28, y1 + 28], outline=magenta, width=5)
    draw.text((650, y1 - 34), "Correct predictions are", fill=black, font=body_font)
    draw.text((650, y1 + 30), "outlined in", fill=black, font=body_font)
    draw.text((842, y1 + 30), "magenta.", fill=magenta, font=body_font)

    y2 = 680
    draw.ellipse([592 - 20, y2 - 20, 592 + 20, y2 + 20], fill=red)
    draw.text((650, y2 - 34), "Incorrect predictions", fill=black, font=body_font)
    draw.text((650, y2 + 30), "are shown in", fill=black, font=body_font)
    draw.text((840, y2 + 30), "red.", fill=red, font=body_font)

    tiny_font = load_font(34)
    draw.text((30, 20), f"Frame {frame_idx}", fill=(60, 60, 60), font=tiny_font)
    draw.text((30, 58), f"Time {frame_idx / FRAME_RATE:.2f}s", fill=(60, 60, 60), font=tiny_font)


def render_frames_split(y_pred: np.ndarray, y_gt_classes: np.ndarray, out_frames_dir: Path, max_frames: int | None = None) -> int:
    prepare_frames_dir(out_frames_dir)
    total = len(y_gt_classes) if max_frames is None else min(len(y_gt_classes), max_frames)
    font = ImageFont.load_default()
    for i in range(total):
        pred_cls = np.argmax(y_pred[i], axis=-1)
        gt_cls = y_gt_classes[i]
        img = Image.new("RGB", (980, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, "RGB")
        draw.font = font
        draw.text((20, 10), f"Frame: {i}  Time: {i / FRAME_RATE:.2f}s", fill=(20, 20, 20))
        draw_single_tab_split(draw, x0=60, y0=40, title="Ground Truth", cls=gt_cls)
        draw_single_tab_split(draw, x0=530, y0=40, title="Prediction", cls=pred_cls)
        img.save(out_frames_dir / f"frame_{i:06d}.png")
    return total


def render_frames_packaged(
    y_pred: np.ndarray,
    y_gt_classes: np.ndarray,
    out_frames_dir: Path,
    max_frames: int | None = None,
    display_frets: int = 12,
) -> int:
    prepare_frames_dir(out_frames_dir)
    total = len(y_gt_classes) if max_frames is None else min(len(y_gt_classes), max_frames)
    display_frets = max(5, min(display_frets, NUM_FRETS))
    for i in range(total):
        pred_cls = np.argmax(y_pred[i], axis=-1)
        gt_cls = y_gt_classes[i]
        img = Image.new("RGB", (1080, 1080), color=(233, 233, 233))
        draw = ImageDraw.Draw(img, "RGB")
        draw_packaged_frame(draw, pred_cls=pred_cls, gt_cls=gt_cls, frame_idx=i, display_frets=display_frets)
        img.save(out_frames_dir / f"frame_{i:06d}.png")
    return total


def make_gif(frames_dir: Path, out_gif: Path, fps: float, frame_count: int) -> None:
    if frame_count < 1:
        raise ValueError("No frames rendered.")
    images = []
    for i in range(frame_count):
        images.append(Image.open(frames_dir / f"frame_{i:06d}.png"))
    duration_ms = int(round(1000.0 / fps))
    images[0].save(
        out_gif,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )
    for img in images:
        img.close()


def maybe_make_mp4(frames_dir: Path, out_mp4: Path, fps: float, audio_wav: Path | None) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and imageio_ffmpeg is not None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if ffmpeg is None:
        return False

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frames_dir / "frame_%06d.png"),
    ]
    if audio_wav is not None and audio_wav.exists():
        cmd += ["-i", str(audio_wav), "-shortest", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)]
    subprocess.run(cmd, check=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render demo tablature visualization for a saved fold.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Saved run directory (contains 0..5 fold dirs).")
    parser.add_argument("--fold", type=int, required=True, help="Validation fold number (0..5).")
    parser.add_argument("--track", type=str, default=None, help="Track stem, e.g. 00_BN1-129-Eb_comp")
    parser.add_argument("--all-tracks", action="store_true", help="Render all tracks in the selected fold.")
    parser.add_argument("--id-csv", type=Path, default=Path("data/spec_repr/id.csv"), help="ID csv used during training.")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("data/GuitarSet/annotation"),
        help="Folder containing GuitarSet .jams annotation files.",
    )
    parser.add_argument(
        "--gt-source",
        type=str,
        choices=["jams", "predictions"],
        default="jams",
        help="Ground-truth source for the left panel.",
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/GuitarSet/audio/audio_mic"), help="Folder containing *_mic.wav files.")
    parser.add_argument("--out-dir", type=Path, default=Path("demos/generated"), help="Output folder.")
    parser.add_argument(
        "--style",
        type=str,
        choices=["packaged", "split"],
        default="packaged",
        help="Visualization style.",
    )
    parser.add_argument(
        "--display-frets",
        type=int,
        default=12,
        help="Number of frets to display in packaged style.",
    )
    parser.add_argument("--gif-fps", type=float, default=10.0, help="GIF frame rate.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick previews.")
    parser.add_argument("--clean-frames", action="store_true", help="Delete intermediate frame PNGs after rendering.")
    return parser.parse_args()


def render_one_track(
    args: argparse.Namespace,
    validation_ids: list[str],
    y_pred: np.ndarray,
    y_gt: np.ndarray,
    track: str,
) -> None:
    y_pred_track, y_gt_track, frame_numbers = load_track_slice(validation_ids, track, y_pred, y_gt)

    if args.gt_source == "jams":
        y_gt_classes = load_gt_from_jams(args.annotation_root, track, frame_numbers)
    else:
        y_gt_classes = np.argmax(y_gt_track, axis=-1)

    stem = f"{args.run_dir.name}_fold{args.fold}_{track}"
    frames_dir = args.out_dir / f"{stem}_frames"
    gif_path = args.out_dir / f"{stem}.gif"
    mp4_path = args.out_dir / f"{stem}.mp4"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.style == "packaged":
        frame_count = render_frames_packaged(
            y_pred_track,
            y_gt_classes,
            frames_dir,
            max_frames=args.max_frames,
            display_frets=args.display_frets,
        )
    else:
        frame_count = render_frames_split(y_pred_track, y_gt_classes, frames_dir, max_frames=args.max_frames)
    make_gif(frames_dir, gif_path, fps=args.gif_fps, frame_count=frame_count)

    audio_wav = args.audio_root / f"{track}_mic.wav"
    made_mp4 = maybe_make_mp4(frames_dir, mp4_path, fps=FRAME_RATE, audio_wav=audio_wav)

    duration_s = frame_count / FRAME_RATE
    print(f"[{track}] Rendered {frame_count} frames ({duration_s:.2f}s) to {frames_dir}")
    if args.max_frames is not None:
        print(f"[{track}] Frame cap active (--max-frames={args.max_frames}).")
    print(f"[{track}] Wrote GIF: {gif_path}")
    if made_mp4:
        print(f"[{track}] Wrote MP4: {mp4_path}")
    else:
        print(f"[{track}] ffmpeg not found; skipped MP4 export.")
        if audio_wav.exists():
            print(f"[{track}] Install ffmpeg to mux audio into MP4.")

    if args.clean_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"[{track}] Deleted intermediate frames folder.")


def main() -> None:
    args = parse_args()

    if args.all_tracks and args.track is not None:
        raise ValueError("Use either --track or --all-tracks, not both.")
    if not args.all_tracks and not args.track:
        raise ValueError("Provide --track, or use --all-tracks.")

    pred_file = args.run_dir / str(args.fold) / "predictions.npz"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")

    with np.load(pred_file, allow_pickle=False) as loaded:
        y_pred = loaded["y_pred"]
        y_gt = loaded["y_gt"]

    if y_pred.shape[-2:] != (NUM_STRINGS, NUM_CLASSES):
        raise ValueError(f"Unexpected prediction shape {y_pred.shape}; expected (*, 6, 21).")

    validation_ids = load_validation_ids(args.id_csv, args.fold)
    tracks = list_tracks_in_ids(validation_ids) if args.all_tracks else [args.track]
    print(f"Rendering {len(tracks)} track(s) from fold {args.fold}...")
    for track in tracks:
        render_one_track(args, validation_ids, y_pred, y_gt, track)


if __name__ == "__main__":
    main()
