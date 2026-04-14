#!/usr/bin/env python
"""
Create animated GIFs from per-iteration plots across an active learning run.

Produces side-by-side AL vs Baseline comparisons for each plot type.

Usage:
    python make_iteration_gifs.py <run_dir> [--fps 2]

Examples:
    python make_iteration_gifs.py active_learning_output
    python make_iteration_gifs.py active_learning_output --fps 1
"""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def get_font(size):
    """Load a TrueType font, falling back to default."""
    for path in [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def find_iteration_dirs(run_dir):
    """Find all iteration_NNN directories, sorted numerically."""
    dirs = sorted(
        [p for p in Path(run_dir).glob("iteration_[0-9]*") if p.is_dir()],
        key=lambda p: int(re.search(r"(\d+)", p.name).group(1)),
    )
    return dirs


def collect_frame_pairs(iter_dirs, pattern_al, pattern_base):
    """Collect matching AL and baseline plot files for each iteration.

    Returns list of (al_path, base_path, iter_num) for iterations where
    both files exist.
    """
    pairs = []
    for d in iter_dirs:
        al_dir = d / "plots" / "al"
        base_dir = d / "plots" / "baseline"
        al_matches = sorted(al_dir.glob(pattern_al)) if al_dir.exists() else []
        base_matches = sorted(base_dir.glob(pattern_base)) if base_dir.exists() else []
        if al_matches and base_matches:
            iter_num = int(re.search(r"(\d+)", d.name).group(1))
            pairs.append((al_matches[0], base_matches[0], iter_num))
    return pairs


def make_side_by_side(img_left, img_right, label_left, label_right, iteration):
    """Combine two images side-by-side with labels and iteration number."""
    # Resize to same height
    h = max(img_left.height, img_right.height)
    if img_left.height != h:
        w = int(img_left.width * h / img_left.height)
        img_left = img_left.resize((w, h), Image.LANCZOS)
    if img_right.height != h:
        w = int(img_right.width * h / img_right.height)
        img_right = img_right.resize((w, h), Image.LANCZOS)

    # Header height for labels
    font_size = max(22, h // 20)
    font = get_font(font_size)
    header_h = font_size + 20
    gap = 10

    total_w = img_left.width + gap + img_right.width
    total_h = header_h + h

    canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw model labels centered above each image
    for label, x_offset, img_w in [
        (label_left, 0, img_left.width),
        (label_right, img_left.width + gap, img_right.width),
    ]:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        x = x_offset + (img_w - tw) // 2
        draw.text((x, 6), label, fill=(0, 0, 0, 255), font=font)

    # Draw iteration number (top-right)
    iter_label = f"Iteration {iteration}"
    bbox = draw.textbbox((0, 0), iter_label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((total_w - tw - 12, 6), iter_label, fill=(100, 100, 100, 255), font=font)

    # Paste images below header
    canvas.paste(img_left, (0, header_h))
    canvas.paste(img_right, (img_left.width + gap, header_h))

    return canvas


def make_gif(frame_pairs, output_path, fps=2):
    """Create a side-by-side animated GIF from AL/baseline frame pairs."""
    if len(frame_pairs) < 2:
        print(f"  Skipping {output_path.name}: only {len(frame_pairs)} frame(s)")
        return

    imgs = []
    for al_path, base_path, iter_num in frame_pairs:
        img_al = Image.open(al_path).convert("RGBA")
        img_base = Image.open(base_path).convert("RGBA")
        combined = make_side_by_side(img_al, img_base, "Active Learning", "Baseline", iter_num)
        imgs.append(combined)

    # Resize all frames to match the first
    target_size = imgs[0].size
    imgs = [img.resize(target_size, Image.LANCZOS) if img.size != target_size else img
            for img in imgs]

    duration_ms = int(1000 / fps)
    imgs[0].save(
        output_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  Saved {output_path} ({len(imgs)} frames, {fps} fps)")


def generate_gifs(run_dir, fps=2, logger=None):
    """Generate side-by-side AL vs Baseline GIFs for a completed run.

    Args:
        run_dir: Path to the active learning output directory.
        fps: Frames per second (default: 2).
        logger: Optional logger (falls back to print).
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    run_dir = Path(run_dir)
    iter_dirs = find_iteration_dirs(run_dir)
    if not iter_dirs:
        log(f"No iteration directories found in {run_dir}, skipping GIF generation")
        return

    log(f"Generating iteration GIFs ({len(iter_dirs)} iterations)...")

    gif_dir = run_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    # (AL glob, Baseline glob, output name)
    plot_types = [
        ("al_input_histograms_*.png", "baseline_input_histograms_*.png", "input_histograms"),
        ("al_target_histogram_*.png", "baseline_target_histogram_*.png", "target_histogram"),
        ("al_parallel_coords_*.png", "baseline_parallel_coords_*.png", "parallel_coords"),
        ("al_new_input_histograms_*.png", "baseline_new_input_histograms_*.png", "new_input_histograms"),
        ("al_new_target_histogram_*.png", "baseline_new_target_histogram_*.png", "new_target_histogram"),
        ("*_true_vs_pred_train.png", "*_true_vs_pred_train.png", "true_vs_pred_train"),
        ("*_true_vs_pred_validation.png", "*_true_vs_pred_validation.png", "true_vs_pred_validation"),
        ("*_hist_true_vs_pred_train.png", "*_hist_true_vs_pred_train.png", "hist_true_vs_pred_train"),
        ("*_hist_true_vs_pred_validation.png", "*_hist_true_vs_pred_validation.png", "hist_true_vs_pred_validation"),
        ("losses_*.png", "losses_*.png", "losses"),
        ("al_candidate_uncertainty_*.png", "baseline_candidate_uncertainty_*.png", "candidate_uncertainty"),
    ]

    for pat_al, pat_base, name in plot_types:
        pairs = collect_frame_pairs(iter_dirs, pat_al, pat_base)
        if pairs:
            gif_path = gif_dir / f"{name}.gif"
            make_gif(pairs, gif_path, fps=fps)

    log(f"GIFs saved to {gif_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side AL vs Baseline GIFs")
    parser.add_argument("run_dir", help="Active learning output directory")
    parser.add_argument("--fps", type=float, default=2,
                        help="Frames per second (default: 2)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        return

    generate_gifs(run_dir, fps=args.fps)


if __name__ == "__main__":
    main()
