import cv2
import os
from pathlib import Path

def extract_screenshots(video_path, output_folder=None, interval_seconds=15):
    """
    Extract screenshots from a video at specified intervals.

    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to save screenshots (default: same as video)
        interval_seconds (int): Time interval between screenshots in seconds
    """

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Set output folder
    if output_folder is None:
        output_folder = os.path.dirname(video_path)

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video information:")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Extracting screenshot every {interval_seconds} seconds")

    # Calculate frame interval
    frame_interval = int(fps * interval_seconds)

    screenshot_count = 0

    for frame_number in range(0, total_frames, frame_interval):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Calculate timestamp
            timestamp = frame_number / fps

            # Generate filename with timestamp
            video_name = Path(video_path).stem
            filename = f"{video_name}_screenshot_{timestamp:.2f}s.jpg"
            output_path = os.path.join(output_folder, filename)

            # Save the screenshot
            cv2.imwrite(output_path, frame)
            print(f"✓ Saved: {filename}")
            screenshot_count += 1
        else:
            print(f"✗ Failed to read frame at position {frame_number}")

    # Release the video capture object
    cap.release()

    print(f"\n✅ Complete! {screenshot_count} screenshots saved to: {output_folder}")
    return screenshot_count

def main():
    """
    Main function to handle user input and run the screenshot extraction.
    """
    print("=== Video Screenshot Extractor ===")
    print("This script will extract screenshots from your video every 15 seconds.\n")

    # Get video path from user
    while True:
        video_path = input("Enter the path to your MP4 video file: ").strip()
        if os.path.exists(video_path):
            break
        print("File not found. Please enter a valid path.")

    # Optional: custom output folder
    output_folder = input("Enter output folder (press Enter for same location as video): ").strip()
    if not output_folder:
        output_folder = None

    # Optional: custom interval
    interval_input = input("Enter interval in seconds (press Enter for default 15): ").strip()
    if interval_input and interval_input.isdigit():
        interval_seconds = int(interval_input)
    else:
        interval_seconds = 15

    print("\nStarting extraction...\n")

    # Run the extraction
    extract_screenshots(video_path, output_folder, interval_seconds)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
