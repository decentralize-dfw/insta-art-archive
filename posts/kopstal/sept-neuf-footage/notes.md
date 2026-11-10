# KARŞIDAN-2_1.mp4 · content lock V2

Replaces the rejected perimeter-only V1. Painting landmarks, physically framed stage references and completed-painting corners define one projective transform for every source frame. The four input corners for all 1772 frames are recorded in the processing report.

Each output frame contains only a geometric resampling of the same original frame. No AI, generative fill, inpainting, clone fill, temporal pixel replacement or still-image substitution. Missing/clipped areas remain empty or retain the defects already baked into the source.

Native dimensions, 30 fps, all frames, duration and original decoded audio are preserved. H.264 CRF17, 25 Mbit/s ceiling, FastStart.

Independent painting-point test: median 0.735 output pixels; 95th percentile of pair medians 0.937 pixels. This is an observed residual, not a zero-motion guarantee. Low-texture/occluded frames use neighboring geometric reference poses; no missing image data is invented.

SHA-256: `c3749307c85b0b6c0cda5664481d698eb9caebc32d92eb7628bcafb3a3943480`

[Per-frame corners, checks and method](../../../processing/canvas-content-lock-v2/REPORT.md).

**Files (10 Nov 2026):** `source.mp4` is the upload above, untouched. `reel.mp4` is the Instagram version made from it in insta.art with `tools/footage.py` (1080×1920 white frame, no audio, 1 s end card, CRF 17).
