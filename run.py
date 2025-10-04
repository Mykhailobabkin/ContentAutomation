import streamlit as st
import cv2
from moviepy import VideoFileClip, concatenate_videoclips


from Resources.AIGateway import OpenAIRequest
from Resources.AIGateway import GeminiRequest

r1 = OpenAIRequest("gpt-4.1")
r2 = GeminiRequest("gemini-2.5-flash")
r3 = GeminiRequest("veo-3.0-generate-001")

video_path = "video.mp4"
image_path = "last_frame.jpg"

def extract_last_frame(video_path, image_path):
    """
    Extract the last frame from a video and save it as an image.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path where the image should be saved (e.g., 'last_frame.jpg')
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("Error: Video has no frames")
        cap.release()
        return False
    
    # Set position to the last frame (frames are 0-indexed)
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    
    # Read the last frame
    ret, frame = cap.read()
    
    if ret:
        # Save the frame as an image
        cv2.imwrite(image_path, frame)
        print(f"Last frame saved to {image_path}")
        cap.release()
        return True
    else:
        print("Error: Could not read the last frame")
        cap.release()
        return False
    
def concatenate_videos(video1_path, video2_path, output_path, method='compose'):
    """
    Concatenate two videos together (one after the other).
    
    Args:
        video1_path (str): Path to the first video file
        video2_path (str): Path to the second video file
        output_path (str): Path where the combined video should be saved
        method (str): 'compose' to handle different sizes/fps, 'chain' for identical videos
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the video clips
        clip1 = VideoFileClip(video1_path)
        clip2 = VideoFileClip(video2_path)
        
        # Concatenate the clips
        final_clip = concatenate_videoclips([clip1, clip2], method=method)
        
        # Write the result to a file
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Clean up
        clip1.close()
        clip2.close()
        final_clip.close()
        
        print(f"Videos successfully concatenated and saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error concatenating videos: {e}")
        return False

def run():

    idea = r2.internet_search(prompt="Generate me one video idea with recent AI news")
    print("Idea Done \n")

    script = r1.send_request(prompt=f"Generate me an audio script for a 16 second video about this idea: {idea}. The script should be exactly 16 seconds long when spoken out loud and should include only text, no audio effects or other descriptions, return only the scripyt, no other text.")
    print("Script Done \n")

    video_prompt = r1.send_request(prompt = f"""
        You are an AI prompt writer specialized in creating continuous video prompts for Google Veo 3 using a "video → screenshot → image-to-video" technique to make longer stories.

        📌 USER INPUT:

        1. Character Descriptions: [Human-like gorilla]

        2. Main Scene Description: [Character sitting behind a taks and saying that: {script}. Make sure to split the audioscript into the neccesary number of scenes. Each scene should be exactly 8 seconds long]

        3. Number of Scenes: [2]

        4. Visual Style: [Ultra realistic 3D animated style]



        🎯 YOUR TASKS:

        1. Create the first scene as a **Text-to-Video prompt** (Scene 1).

        2. Create follow-up scenes as **Image-to-Video prompts** (Scene 2 → final).

        3. Keep **character descriptions, setting, and color palette consistent** across all scenes.

        4. Always start follow-up scenes with: "continuation of the previous scene".

        5. Add small but clear scene progression in each step (new action, expression, or camera movement).

        6. Maintain the same visual style across all scenes.

        7. When returning the prompts use the following format: 
        Scene 1 (Text-to-Video):
        [Your prompt here]
        
        ///


        Scene 2 (Image-to-Video):
        [Your prompt here]
        """
    )

    print(video_prompt + "\n")

    video_prompt_1 = video_prompt.split("///")[0].replace("Scene 1 (Text-to-Video):", "").strip()
    video_prompt_2 = video_prompt.split("///")[1].replace("Scene 2 (Image-to-Video):", "").strip()




    r3.generate_video(prompt=video_prompt_1)
    extract_last_frame(video_path, image_path)


    r3.generate_video_w_images(prompt=video_prompt_2, image_path=image_path)
    concatenate_videos("video.mp4", "video1.mp4", "final_video.mp4", method='compose')
    print("Videos Done")

if __name__ == "__main__":
    run()