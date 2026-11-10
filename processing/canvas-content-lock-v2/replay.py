"""Replay recorded canvas geometry; source originals are never modified.

Examples:
  python replay.py INPUT.mp4 board-game-corners.csv OUTPUT.mp4 --rect 500 130 1300 1040 --source-matte
  python replay.py INPUT.mp4 sept-neuf-corners.csv OUTPUT.mp4 --rect 142 30 616 840

Requires numpy, opencv-python-headless and ffmpeg on PATH.
"""
import argparse
from pathlib import Path
import subprocess
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('corners', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rect', type=int, nargs=4, required=True, metavar=('X','Y','WIDTH','HEIGHT'))
    parser.add_argument('--source-matte', action='store_true', help='Preserve original Board game intro/matte brightness')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must differ from the original input')
    cv2.setNumThreads(4)
    rows = np.loadtxt(args.corners, delimiter=',', skiprows=1)
    quads = rows[:, 1:].reshape(-1, 4, 2).astype(np.float32)
    cap = cv2.VideoCapture(str(args.input))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if len(quads) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        raise ValueError('Corner count does not match source frames')
    x, y, w, h = args.rect
    if min(x,y) < 0 or x+w > width or y+h > height:
        raise ValueError('Destination rectangle falls outside the original frame')
    destination = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    command = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
               '-s',f'{width}x{height}','-r',str(fps),'-i','-','-i',str(args.input),
               '-map','0:v:0','-map','1:a:0?','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709',
               '-c:v','libx264','-preset','slow','-crf','17','-maxrate','25M','-bufsize','25M',
               '-threads','4','-pix_fmt','yuv420p','-c:a','copy','-color_range','tv','-colorspace','bt709',
               '-color_trc','bt709','-color_primaries','bt709','-movflags','+faststart',str(args.output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for quad in quads:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Source ended before all recorded corners were used')
            matrix = cv2.getPerspectiveTransform(quad, destination)
            patch = cv2.warpPerspective(frame, matrix, (w,h), flags=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            result = np.full_like(frame, 255)
            if args.source_matte:
                sample = np.concatenate([frame[10,::80],frame[-10,::80]], axis=0)
                result[:] = np.median(sample,axis=0).astype(np.uint8)
            result[y:y+h,x:x+w] = patch
            encoder.stdin.write(result.tobytes())
    finally:
        cap.release()
        encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
