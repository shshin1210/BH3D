import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEOS = Path("static/videos")
# NAMES = ["powder", "water", "money", "vinyl", "hand"]
NAMES = ["BH3D_fast_forward_imaging_system"]

def reencode(src: Path, dst: Path) -> None:
    cmd = [
        FFMPEG, "-y", "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-crf", "20",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.STDOUT)


def main() -> int:
    for name in NAMES:
        src = VIDEOS / f"{name}.mp4"
        backup = VIDEOS / f"{name}_orig.mp4"
        tmp = VIDEOS / f"{name}_h264.mp4"

        if not src.exists():
            print(f"[skip] {src} not found")
            continue

        if not backup.exists():
            src.rename(backup)
            print(f"[backup] {src.name} -> {backup.name}")

        reencode(backup, tmp)
        tmp.rename(src)
        print(f"[done]   {src.name}  ({backup.stat().st_size:,} -> {src.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
