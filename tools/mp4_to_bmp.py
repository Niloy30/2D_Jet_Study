import os
import subprocess

# Set your MP4 video path here
video_path = r"C:\Users\niloy\Desktop\Experiments\05082025\20250508_173653.mp4"
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Update if needed

def convert_mp4_to_bmp(video_path):
    if not os.path.isfile(video_path) or not video_path.lower().endswith(".mp4"):
        print("Invalid file path or not an .mp4 file.")
        return

    # Extract base name and parent directory
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    parent_dir = os.path.dirname(video_path)
    
    # Create output folder named after the video
    output_folder = os.path.join(parent_dir, base_name)
    os.makedirs(output_folder, exist_ok=True)

    # Output file pattern: <video_name><frame_number>.bmp
    output_pattern = os.path.join(output_folder, f"{base_name}%06d.bmp")

    # Construct and run ffmpeg command
    command = [
        ffmpeg_path,
        "-i", video_path,
        output_pattern
    ]

    print(f"Converting {video_path} to BMP frames...")
    subprocess.run(command, check=True)
    print(f"Frames saved in: {output_folder}")

if __name__ == "__main__":
    convert_mp4_to_bmp(video_path)
