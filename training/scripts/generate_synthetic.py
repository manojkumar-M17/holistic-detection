#!/usr/bin/env python3
"""
Deterministic Synthetic Exam Scene Generator
--------------------------------------------
Generates synthetic exam desk environments and objects using procedural
geometric synthesis (OpenCV + NumPy) without requiring human volunteers.

Objects Supported (Canonical Taxonomy):
  0: person
  1: phone
  2: paper
  3: book
  4: calculator
  5: laptop
  6: watch
  7: earphone_or_earbud
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class SyntheticExamSceneGenerator:
    """Procedurally synthesizes exam hall desk scenes and objects."""

    def __init__(self, seed: int = 42, width: int = 640, height: int = 480):
        self.seed = seed
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)

    def _safe_randint(self, low: int, high: int) -> int:
        if low >= high:
            return int(low)
        return int(self.rng.randint(low, high))

    def _render_background(self) -> np.ndarray:
        """Create a textured desk and classroom wall background."""
        # Top 30-40% is wall/room background, bottom 60-70% is desk
        desk_y = int(self.height * self.rng.uniform(0.30, 0.42))
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Wall color (light grey, cream, or muted blue/beige)
        wall_color = np.array([
            self.rng.randint(180, 230),
            self.rng.randint(180, 230),
            self.rng.randint(180, 230)
        ], dtype=np.uint8)
        img[:desk_y, :] = wall_color

        # Subtle vertical wall texture
        wall_noise = self.rng.randint(-8, 8, (desk_y, self.width, 3)).astype(np.int16)
        img[:desk_y, :] = np.clip(img[:desk_y, :].astype(np.int16) + wall_noise, 0, 255).astype(np.uint8)

        # Desk type: 0 = Wood grain, 1 = Grey office laminate, 2 = Dark desk
        desk_type = self.rng.choice([0, 1, 2], p=[0.45, 0.40, 0.15])
        if desk_type == 0:  # Wood grain
            base_desk = np.array([
                self.rng.randint(30, 70),   # B
                self.rng.randint(70, 120),  # G
                self.rng.randint(130, 180), # R
            ], dtype=np.uint8)
        elif desk_type == 1:  # Grey laminate
            g = self.rng.randint(140, 190)
            base_desk = np.array([g, g, g], dtype=np.uint8)
        else:  # Dark desk
            d = self.rng.randint(45, 80)
            base_desk = np.array([d, d, d], dtype=np.uint8)

        desk_h = self.height - desk_y
        desk_region = np.full((desk_h, self.width, 3), base_desk, dtype=np.uint8)

        # Add horizontal grain / gradient
        grad = np.linspace(0.85, 1.15, desk_h)[:, None, None]
        desk_region = np.clip(desk_region.astype(np.float32) * grad, 0, 255).astype(np.uint8)

        # Procedural surface noise
        noise = self.rng.normal(0, 5, (desk_h, self.width, 3)).astype(np.int16)
        desk_region = np.clip(desk_region.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Desk edge shadow
        cv2.line(desk_region, (0, 0), (self.width, 0), (30, 30, 30), 2)

        img[desk_y:, :] = desk_region
        return img

    def _render_person(self, img: np.ndarray) -> Optional[Tuple[int, float, float, float, float]]:
        """Render a synthetic student silhouette (torso, head, shoulders)."""
        # Centered or slightly off-center
        cx = int(self.width * self.rng.uniform(0.42, 0.58))
        head_cy = int(self.height * self.rng.uniform(0.24, 0.35))
        head_r = int(self.width * self.rng.uniform(0.065, 0.085))

        # Skin tone
        skin_color = (
            int(self.rng.randint(110, 170)),
            int(self.rng.randint(140, 200)),
            int(self.rng.randint(180, 240)),
        )
        # Shirt color
        shirt_color = (
            int(self.rng.randint(40, 200)),
            int(self.rng.randint(40, 200)),
            int(self.rng.randint(40, 200)),
        )
        # Hair color
        hair_color = (
            int(self.rng.randint(15, 50)),
            int(self.rng.randint(15, 50)),
            int(self.rng.randint(15, 50)),
        )

        # Torso / shoulders (trapezoid / polygon)
        shoulder_w = int(head_r * self.rng.uniform(3.0, 4.0))
        shoulder_y = head_cy + int(head_r * 0.9)
        bottom_y = int(self.height * 0.75)
        bottom_w = int(shoulder_w * 1.3)

        torso_pts = np.array([
            [cx - shoulder_w // 2, shoulder_y],
            [cx + shoulder_w // 2, shoulder_y],
            [cx + bottom_w // 2, bottom_y],
            [cx - bottom_w // 2, bottom_y],
        ], dtype=np.int32)
        cv2.fillPoly(img, [torso_pts], shirt_color)

        # Neck
        neck_w = int(head_r * 0.6)
        neck_pts = np.array([
            [cx - neck_w // 2, head_cy],
            [cx + neck_w // 2, head_cy],
            [cx + neck_w // 2, shoulder_y],
            [cx - neck_w // 2, shoulder_y],
        ], dtype=np.int32)
        cv2.fillPoly(img, [neck_pts], skin_color)

        # Head (ellipse)
        cv2.ellipse(img, (cx, head_cy), (head_r, int(head_r * 1.25)), 0, 0, 360, skin_color, -1)

        # Hair
        cv2.ellipse(img, (cx, head_cy - int(head_r * 0.3)), (head_r + 2, int(head_r * 0.9)), 0, 180, 360, hair_color, -1)

        # Compute bounding box
        x_min = max(0, cx - bottom_w // 2)
        x_max = min(self.width, cx + bottom_w // 2)
        y_min = max(0, head_cy - int(head_r * 1.25))
        y_max = min(self.height, bottom_y)

        xc = (x_min + x_max) / (2.0 * self.width)
        yc = (y_min + y_max) / (2.0 * self.height)
        w = (x_max - x_min) / float(self.width)
        h = (y_max - y_min) / float(self.height)
        return (0, xc, yc, w, h)

    def _render_phone(self, img: np.ndarray) -> Optional[Tuple[int, float, float, float, float]]:
        """Render a smartphone lying flat or tilted on desk."""
        pw = int(self.width * self.rng.uniform(0.06, 0.10))
        ph = int(pw * self.rng.uniform(1.8, 2.2))
        px = self._safe_randint(int(self.width * 0.1), int(self.width * 0.9 - pw))
        py = self._safe_randint(int(self.height * 0.55), int(self.height * 0.92 - ph))

        # Body (dark slate or silver)
        body_col = (
            self.rng.randint(20, 50),
            self.rng.randint(20, 50),
            self.rng.randint(20, 50),
        )
        cv2.rectangle(img, (px, py), (px + pw, py + ph), body_col, -1)
        # Bezel highlight
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (80, 80, 80), 1)

        # Screen (black, or subtle glowing OLED screen)
        screen_on = self.rng.choice([True, False], p=[0.7, 0.3])
        margin = max(2, int(pw * 0.08))
        if screen_on:
            screen_col = (
                self.rng.randint(180, 240),
                self.rng.randint(180, 240),
                self.rng.randint(180, 255),
            )
            cv2.rectangle(img, (px + margin, py + margin), (px + pw - margin, py + ph - margin), screen_col, -1)
            # Simulated screen content / app lines
            for ly in range(py + margin + 8, py + ph - margin - 8, 8):
                cv2.line(img, (px + margin + 4, ly), (px + pw - margin - 4, ly), (60, 60, 60), 1)
        else:
            cv2.rectangle(img, (px + margin, py + margin), (px + pw - margin, py + ph - margin), (15, 15, 15), -1)

        xc = (px + pw / 2.0) / self.width
        yc = (py + ph / 2.0) / self.height
        w = pw / float(self.width)
        h = ph / float(self.height)
        return (1, xc, yc, w, h)

    def _render_paper(self, img: np.ndarray) -> Optional[Tuple[int, float, float, float, float]]:
        """Render a sheet of exam paper / answer sheet."""
        pw = int(self.width * self.rng.uniform(0.18, 0.28))
        ph = int(pw * self.rng.uniform(1.2, 1.45))
        px = self._safe_randint(int(self.width * 0.08), int(self.width * 0.92 - pw))
        py = self._safe_randint(int(self.height * 0.45), int(self.height * 0.94 - ph))

        # Paper color (white or slight ivory)
        paper_col = (
            self.rng.randint(235, 250),
            self.rng.randint(238, 255),
            self.rng.randint(240, 255),
        )
        cv2.rectangle(img, (px, py), (px + pw, py + ph), paper_col, -1)
        # Drop shadow underneath / border
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (180, 180, 180), 1)

        # Simulated ruled lines or printed text
        line_spacing = max(6, int(ph * 0.06))
        for ly in range(py + line_spacing * 2, py + ph - line_spacing, line_spacing):
            line_w = self._safe_randint(int(pw * 0.6), int(pw * 0.9))
            cv2.line(img, (px + int(pw * 0.08), ly), (px + line_w, ly), (160, 160, 170), 1)

        xc = (px + pw / 2.0) / self.width
        yc = (py + ph / 2.0) / self.height
        w = pw / float(self.width)
        h = ph / float(self.height)
        return (2, xc, yc, w, h)

    def _render_book(self, img: np.ndarray) -> Optional[Tuple[int, float, float, float, float]]:
        """Render an open or closed textbook / notebook."""
        is_open = self.rng.choice([True, False], p=[0.6, 0.4])
        if is_open:
            bw = int(self.width * self.rng.uniform(0.24, 0.36))
            bh = int(bw * self.rng.uniform(0.65, 0.80))
            bx = self._safe_randint(int(self.width * 0.05), int(self.width * 0.95 - bw))
            by = self._safe_randint(int(self.height * 0.48), int(self.height * 0.94 - bh))

            # Left and right pages
            page_col = (self.rng.randint(230, 245), self.rng.randint(235, 250), self.rng.randint(240, 255))
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), page_col, -1)
            # Center spine shadow
            spine_x = bx + bw // 2
            cv2.line(img, (spine_x, by), (spine_x, by + bh), (120, 120, 120), 2)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)
        else:
            bw = int(self.width * self.rng.uniform(0.16, 0.24))
            bh = int(bw * self.rng.uniform(1.2, 1.4))
            bx = self._safe_randint(int(self.width * 0.05), int(self.width * 0.95 - bw))
            by = self._safe_randint(int(self.height * 0.50), int(self.height * 0.94 - bh))

            cover_col = (
                self.rng.randint(40, 180),
                self.rng.randint(40, 180),
                self.rng.randint(40, 180),
            )
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), cover_col, -1)
            # Spine edge
            cv2.line(img, (bx, by), (bx, by + bh), (30, 30, 30), 3)

        xc = (bx + bw / 2.0) / self.width
        yc = (by + bh / 2.0) / self.height
        w = bw / float(self.width)
        h = bh / float(self.height)
        return (3, xc, yc, w, h)

    def _render_calculator(self, img: np.ndarray) -> Optional[Tuple[int, float, float, float, float]]:
        """Render a scientific or standard calculator."""
        cw = int(self.width * self.rng.uniform(0.08, 0.13))
        ch = int(cw * self.rng.uniform(1.6, 2.0))
        cx = self._safe_randint(int(self.width * 0.08), int(self.width * 0.92 - cw))
        cy = self._safe_randint(int(self.height * 0.52), int(self.height * 0.93 - ch))

        body_col = (
            self.rng.randint(50, 80),
            self.rng.randint(50, 80),
            self.rng.randint(50, 80),
        )
        cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), body_col, -1)
        cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (100, 100, 100), 1)

        # LCD screen at top
        screen_h = max(8, int(ch * 0.18))
        margin = max(3, int(cw * 0.08))
        lcd_col = (self.rng.randint(140, 170), self.rng.randint(160, 190), self.rng.randint(150, 180))
        cv2.rectangle(img, (cx + margin, cy + margin), (cx + cw - margin, cy + margin + screen_h), lcd_col, -1)

        # Keypad matrix (dots/rectangles)
        btn_start_y = cy + margin + screen_h + 4
        rows, cols = 4, 3
        btn_w = (cw - 2 * margin) // cols
        btn_h = (cy + ch - btn_start_y - margin) // rows
        for r in range(rows):
            for c in range(cols):
                bx1 = cx + margin + c * btn_w + 1
                by1 = btn_start_y + r * btn_h + 1
                bx2 = bx1 + btn_w - 2
                by2 = by1 + btn_h - 2
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (30, 30, 30), -1)

        xc = (cx + cw / 2.0) / self.width
        yc = (cy + ch / 2.0) / self.height
        w = cw / float(self.width)
        h = ch / float(self.height)
        return (4, xc, yc, w, h)

    def _render_laptop(self, img: np.ndarray) -> Optional[Tuple[int, float, float, float, float]]:
        """Render a laptop on the desk."""
        lw = int(self.width * self.rng.uniform(0.30, 0.42))
        lh = int(lw * self.rng.uniform(0.55, 0.70))
        lx = self._safe_randint(int(self.width * 0.05), int(self.width * 0.95 - lw))
        ly = self._safe_randint(int(self.height * 0.42), int(self.height * 0.90 - lh))

        base_col = (
            self.rng.randint(160, 190),
            self.rng.randint(160, 190),
            self.rng.randint(160, 190),
        )
        # Clamshell top / screen (upper half)
        screen_h = int(lh * 0.55)
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + screen_h), (40, 40, 40), -1)
        # Display panel
        disp_margin = max(4, int(lw * 0.04))
        disp_col = (self.rng.randint(180, 220), self.rng.randint(180, 220), self.rng.randint(200, 240))
        cv2.rectangle(img, (lx + disp_margin, ly + disp_margin), (lx + lw - disp_margin, ly + screen_h - 2), disp_col, -1)

        # Base / keyboard (lower half)
        cv2.rectangle(img, (lx - 4, ly + screen_h), (lx + lw + 4, ly + lh), base_col, -1)
        # Trackpad
        tw = int(lw * 0.25)
        th = int(lh * 0.15)
        tx = lx + (lw - tw) // 2
        ty = ly + lh - th - 3
        cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (120, 120, 120), 1)

        xc = (lx + lw / 2.0) / self.width
        yc = (ly + lh / 2.0) / self.height
        w = lw / float(self.width)
        h = lh / float(self.height)
        return (5, xc, yc, w, h)

    def _render_watch(self, img: np.ndarray) -> Optional[Tuple[int, float, float, float, float]]:
        """Render a wristwatch / smartwatch on desk or student."""
        ww = int(self.width * self.rng.uniform(0.04, 0.07))
        wh = int(ww * self.rng.uniform(1.4, 2.0))
        wx = self._safe_randint(int(self.width * 0.1), int(self.width * 0.9 - ww))
        wy = self._safe_randint(int(self.height * 0.55), int(self.height * 0.92 - wh))

        # Strap
        strap_w = max(4, int(ww * 0.5))
        strap_x = wx + (ww - strap_w) // 2
        cv2.rectangle(img, (strap_x, wy), (strap_x + strap_w, wy + wh), (30, 30, 35), -1)

        # Watch body / dial in center
        dial_cy = wy + wh // 2
        dial_r = ww // 2
        cv2.circle(img, (wx + dial_r, dial_cy), dial_r, (180, 180, 190), -1)
        # Dial face
        cv2.circle(img, (wx + dial_r, dial_cy), max(2, dial_r - 2), (20, 20, 20), -1)

        xc = (wx + ww / 2.0) / self.width
        yc = (wy + wh / 2.0) / self.height
        w = ww / float(self.width)
        h = wh / float(self.height)
        return (6, xc, yc, w, h)

    def _render_earphone(self, img: np.ndarray) -> Optional[Tuple[int, float, float, float, float]]:
        """Render wireless earbud case or earbud on desk."""
        ew = int(self.width * self.rng.uniform(0.04, 0.08))
        eh = int(ew * self.rng.uniform(0.8, 1.2))
        ex = self._safe_randint(int(self.width * 0.1), int(self.width * 0.9 - ew))
        ey = self._safe_randint(int(self.height * 0.55), int(self.height * 0.92 - eh))

        case_col = (
            self.rng.randint(230, 255),
            self.rng.randint(230, 255),
            self.rng.randint(230, 255),
        )
        # Rounded earbud charging case
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), case_col, -1)
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (170, 170, 170), 1)
        # Seam line
        cv2.line(img, (ex, ey + eh // 2), (ex + ew, ey + eh // 2), (150, 150, 150), 1)

        xc = (ex + ew / 2.0) / self.width
        yc = (ey + eh / 2.0) / self.height
        w = ew / float(self.width)
        h = eh / float(self.height)
        return (7, xc, yc, w, h)

    def generate_scene(self, allow_empty: bool = False) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """Generate a complete exam desk scene with labels."""
        img = self._render_background()
        annotations: List[Tuple[int, float, float, float, float]] = []

        # 90% of scenes have a person (candidate sitting at desk)
        has_person = self.rng.choice([True, False], p=[0.90, 0.10])
        if has_person:
            person_box = self._render_person(img)
            if person_box:
                annotations.append(person_box)

        # If not allow_empty or random decision: place 1-4 objects
        if not allow_empty or self.rng.random() > 0.15:
            # Pick objects to spawn (probabilities favor paper, phone, book, calculator)
            possible_renderers = [
                (self._render_paper, 0.30),
                (self._render_phone, 0.25),
                (self._render_book, 0.15),
                (self._render_calculator, 0.12),
                (self._render_watch, 0.08),
                (self._render_earphone, 0.06),
                (self._render_laptop, 0.04),
            ]
            
            num_objects = self.rng.randint(1, 4)
            funcs, weights = zip(*possible_renderers)
            weights = np.array(weights) / sum(weights)

            selected_renderers = self.rng.choice(funcs, size=num_objects, p=weights, replace=True)
            for renderer in selected_renderers:
                box = renderer(img)
                if box:
                    annotations.append(box)

        # Apply slight camera blur / sensor noise / lighting variation
        if self.rng.random() > 0.4:
            kernel_size = self.rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Slight brightness adjustment
        brightness = self.rng.uniform(0.9, 1.1)
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        return img, annotations


def generate_synthetic_dataset(
    output_dir: Path,
    count: int = 50,
    seed: int = 42,
    width: int = 640,
    height: int = 480,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticExamSceneGenerator(seed=seed, width=width, height=height)
    
    class_counts: Dict[int, int] = {}
    manifest_records: List[Dict[str, Any]] = []

    for idx in range(count):
        img_name = f"synth_{idx:06d}.jpg"
        lbl_name = f"synth_{idx:06d}.txt"

        img, annotations = generator.generate_scene(allow_empty=allow_empty)

        # Save image
        cv2.imwrite(str(images_dir / img_name), img)

        # Save label
        lbl_path = labels_dir / lbl_name
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cls_id, xc, yc, w, h in annotations:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        manifest_records.append({
            "image": img_name,
            "label": lbl_name,
            "annotations_count": len(annotations),
            "classes": [a[0] for a in annotations],
        })

    metadata = {
        "dataset_type": "synthetic_exam_scenes",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_images": count,
        "width": width,
        "height": height,
        "class_counts": class_counts,
        "canonical_taxonomy": {
            0: "person",
            1: "phone",
            2: "paper",
            3: "book",
            4: "calculator",
            5: "laptop",
            6: "watch",
            7: "earphone_or_earbud",
        },
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic synthetic exam hall scene generator.")
    parser.add_argument("--output-dir", type=str, default="training/datasets/synthetic", help="Output directory")
    parser.add_argument("--count", type=int, default=50, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--allow-empty", action="store_true", help="Allow some scenes without suspicious objects")

    args = parser.parse_args()
    out_path = Path(args.output_dir)

    print(f"Generating {args.count} synthetic scenes with seed {args.seed} into {out_path}...")
    metadata = generate_synthetic_dataset(
        output_dir=out_path,
        count=args.count,
        seed=args.seed,
        width=args.width,
        height=args.height,
        allow_empty=args.allow_empty,
    )
    print(f"Generation complete: {metadata['total_images']} images created.")
    print(f"Class distribution: {metadata['class_counts']}")
    print(f"Metadata saved to: {out_path / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
