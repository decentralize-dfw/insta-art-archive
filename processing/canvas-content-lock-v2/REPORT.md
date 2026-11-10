# Painting-content stabilization · V2

This replaces the rejected V1 which fixed the outside matte while painting content still shifted. Only the two Kopstal footage derivatives are updated. Capricorn sisters and Viperella remain byte-identical to the previously uploaded originals.

| File | Frames | Duration | Native dimensions | Independent median residual | P95 of pair medians |
| --- | ---: | ---: | --- | ---: | ---: |
| [board-game](../../posts/kopstal/7-fam-board-game-footage/reel.mp4) | 823 | 27.433333 s | 2300 × 1300 | 1.057 px | 2.263 px |
| [sept-neuf](../../posts/kopstal/sept-neuf-footage/reel.mp4) | 1772 | 59.066667 s | 900 × 900 | 0.735 px | 0.937 px |

## Geometry and evidence

The completed painting's physical four-corner plane is the reference. Prepared full-canvas stage images anchor scale and perspective throughout the process. Nearby real video reference frames supply stable paint landmarks. A robust projective graph relates these references, and each original frame is registered directly to nearby references. Spatially distributed inliers reject isolated hand/brush motion. Board game's weak early frames and selected transitions additionally use painting-area feature registration at source resolution and a short local reference graph for frames 112–146. These operations estimate coordinates only.

Every input frame has its own recorded source quadrilateral (TL, TR, BR, BL). The source-to-canvas matrices and corner CSVs are supplied. Mapping these corners to the destination is an algebraic identity; it does not prove zero real camera motion. Independent SIFT correspondences at held-out frame pairs measure actual painting content after transformation. Source compression, blur, low texture, changing paint and rolling/nonplanar distortion limit subpixel certainty. The JSON reports include worst measured pairs rather than hiding them.

Board game uses destination (500,130,1300,1040) in the original 2300×1300 raster; Sept uses (142,30,616,840) in 900×900. In low-texture or occluded frames that cannot support a trustworthy match, neighboring geometric reference poses are used and explicitly flagged in the per-frame records. The painting pixels are never temporally interpolated.

## No image completion

Each output is a single perspective resampling of that same source frame. No AI, inpainting, generative extension, clone filling, reconstructed missing painting, frozen frame or substituted photograph. Black/blank/clipped source regions are retained; out-of-source samples remain empty black. The exterior presentation matte is separate from the painting. Neither input original is overwritten.

All source frames, native raster dimensions, 30 fps, durations and decoded audio hashes are preserved. The MERGED-20sec source actually lasts 27.433333 seconds. AAC is stream-copied. Both final files fully decode and use MP4 FastStart and HTTP byte-range playback in the local viewer.

## Reproduction

`replay.py` takes the original video and the matching `*-four-corners-v2.csv`. Examples:

```
python replay.py INPUT.mp4 board-game-four-corners-v2.csv OUTPUT.mp4 --rect 500 130 1300 1040 --source-matte
python replay.py INPUT.mp4 sept-neuf-four-corners-v2.csv OUTPUT.mp4 --rect 142 30 616 840
```

Requires numpy, OpenCV and ffmpeg. The tracked coordinate data are sufficient to reproduce the geometry without repeating feature detection. `manifest.json` records source and output hashes, frame counts, audio checks and low-texture frame IDs. `method/` contains the classical computer-vision estimation and validation code used for this correction. Local source paths are resolved relative to the archive workspace; no credentials or external services are involved.
