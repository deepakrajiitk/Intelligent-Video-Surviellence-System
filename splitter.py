from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Set the input video file path
input_file = "box3.avi"

# Set the output video file path
output_file = "box.avi"

# Set the start and end time for the desired segment in seconds
start_time = 1 * 60  # 10 minutes
end_time = 6 * 60  # 15 minutes

# Extract the desired segment using moviepy's ffmpeg_extract_subclip function
ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)
