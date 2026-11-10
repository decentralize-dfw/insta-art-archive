# KARŞIDAN-2_1.mp4

Perspective-corrected derivative. Original source unchanged on disk.

Canvas image is mapped into the same fixed rectangle in every frame: [142, 30, 616, 840] (x, y, width, height). Corner mapping error is numerical transform precision, not a claim of zero physical feature-tracking error.

Source duration, 1772 video frames, 900 × 900 output and 30 fps are preserved. Audio stream is copied; decoded audio SHA-256 matches the source. No generative fill, colour grading, speed changes, missing-frame replacement or still-image substitution.

The final geometry uses direct robust outside-edge fits for each frame, with no accumulated feature pose. A three-frame median filters corner measurements only. Preliminary surface-feature tracking was rejected because the moving hand could be mistaken for camera movement.

Source SHA-256: `c79e8ae0cca7a13cb412dfcb097741384753f6b609e19997b81c9cca58f0aa57`

Archive SHA-256: `86bac75e96e390a864f0d45522fff0c763f5664470454a05470c8a152c87d84d`

Validation and per-frame geometry: [processing report](../../../processing/canvas-lock-2026-11-10/REPORT.md).
