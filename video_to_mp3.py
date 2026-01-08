'''# Converts the videos to mp3 
import os 
import subprocess

# Create the audios directory if it doesn't exist
os.makedirs("audios", exist_ok=True)

files = os.listdir("videos") 
for file in files:

    # ✅ Skip non-video files (like .webp)
    if not file.lower().endswith((".mp4", ".mkv", ".webm", ".avi")):
        continue

    # ✅ Ensure '#' exists before splitting
    if "#" not in file:
        continue

    tutorial_number = file.split(" [")[0].split("#")[1].strip()
    file_name = file.split(" ｜ ")[0]

    print(tutorial_number, file_name)

    subprocess.run([
        "ffmpeg",
        "-i", f"videos/{file}",
        f"audios/{tutorial_number}_{file_name}.mp3"
    ])
'''
# Converts the videos to mp3
import os
import subprocess

os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")

for file in files:

    # Skip non-video files
    if not file.lower().endswith((".mp4", ".mkv", ".webm", ".avi")):
        continue

    if "#" not in file:
        continue

    tutorial_number = file.split(" [")[0].split("#")[1].strip()
    file_name = file.split(" ｜ ")[0]

    print(f"Processing: {tutorial_number} - {file_name}")

    subprocess.run([
        "ffmpeg",
        "-y",                         # overwrite output
        "-err_detect", "ignore_err",  # ignore corrupted opus packets
        "-i", f"videos/{file}",
        "-vn",                        # remove video
        "-acodec", "libmp3lame",      # force mp3 encoder
        "-ab", "128k",                # bitrate
        f"audios/{tutorial_number}_{file_name}.mp3"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
