import streamlit as st
import cv2
from moviepy import VideoFileClip, concatenate_videoclips
import os


from Resources.AIResource import OpenAIRequest
from Resources.AIResource import GeminiRequest

r1 = OpenAIRequest("gpt-4.1")
r2 = GeminiRequest("gemini-2.5-flash")
r3 = GeminiRequest("veo-3.0-fast-generate-001")

number_of_scenes = 3

animal = "Elephant"

seconds = (number_of_scenes*8) - 4

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

    script = r1.send_request(prompt=f"Generate me an audio script for a {seconds} second video about this idea: {idea}. The script should be exactly 16 seconds long when spoken out loud and should include only text, no audio effects or other descriptions, return only the scripyt, no other text.")
    print("Script Done \n")

    video_prompt = r1.send_request(prompt = f"""
                                   
        You are an AI prompt writer specialized in creating continuous video prompts for Google Veo 3 using a "video → screenshot → image-to-video" technique to make longer stories.

        📌 USER INPUT:

        1. Character Descriptions: [{animal}, with a masculine and clear voice]

        2. Main Scene Description: [Character sitting behind a taks and saying that: {script}. Make sure to split the audioscript into 2 scenes evenly, so the video won't break in a middle of a word. Each scene should be exactly 8 seconds long]

        3. Number of Scenes: [{number_of_scenes}]

        4. Visual Style: [Ultra realistic 3D animated style]



        🎯 YOUR TASKS:

        1. Create the first scene as a **Text-to-Video prompt** (Scene 1).

        2. Create follow-up scenes as **Image-to-Video prompts** (Scene 2 → final).

        3. Keep **character descriptions, setting, and color palette consistent** across all scenes.

        4. Add small but clear scene progression in each step (new action, expression, or camera movement).

        5. Maintain the same visual style across all scenes.

        6. When returning the prompts use the following format: 
        Scene 1 (Text-to-Video):
        [Your prompt here]
        
        ///


        Scene 2 (Image-to-Video):
        [Your prompt here]


        📌 Prompt writing basics
        Good prompts are descriptive and clear. To get the most out of Veo, start with identifying your core idea, refine your idea by adding keywords and modifiers, and incorporate video-specific terminology into your prompts.

        The following elements should be included in your prompt:

        Subject: The object, person, animal, or scenery that you want in your video, such as cityscape, nature, vehicles, or puppies.
        Action: What the subject is doing (for example, walking, running, or turning their head).
        Style: Specify creative direction using specific film style keywords, such as sci-fi, horror film, film noir, or animated styles like cartoon.
        Camera positioning and motion: [Optional] Control the camera's location and movement using terms like aerial view, eye-level, top-down shot, dolly shot, or worms eye.
        Composition: [Optional] How the shot is framed, such as wide shot, close-up, single-shot or two-shot.
        Focus and lens effects: [Optional] Use terms like shallow focus, deep focus, soft focus, macro lens, and wide-angle lens to achieve specific visual effects.
        Ambiance: [Optional] How the color and light contribute to the scene, such as blue tones, night, or warm tones.
        More tips for writing prompts
        Use descriptive language: Use adjectives and adverbs to paint a clear picture for Veo.
        Enhance the facial details: Specify facial details as a focus of the photo like using the word portrait in the prompt.

        """
    )

    video_prompt_1 = video_prompt.split("///")[0].replace("Scene 1 (Text-to-Video):", "").strip()
    video_prompt_2 = video_prompt.split("///")[1].replace("Scene 2 (Image-to-Video):", "").strip()

    video_prompt_3 = video_prompt.split("///")[2].replace("Scene 3 (Image-to-Video):", "").strip()

    print(f"\n{video_prompt_1} \n\n {video_prompt_2} \n\n {video_prompt_3} \n")


    r3.generate_video(prompt=video_prompt_1)
    print("Video 1 Done \n")
    extract_last_frame(video_path, image_path)

    r3.generate_video_w_images(prompt=video_prompt_2, image_path=image_path, output_path="video1.mp4")
    print("Video 2 Done \n")
    extract_last_frame(video_path, image_path)

    concatenate_videos("video.mp4", "video1.mp4", "2_video.mp4", method='compose')
    os.remove("video.mp4")
    os.remove("video1.mp4")
    print("2 videos Done")

    r3.generate_video_w_images(prompt=video_prompt_3, image_path=image_path, output_path="video2.mp4")
    print("Video 3 Done")

    concatenate_videos("2_video", "video2.mp4", "final_video.mp4", method='compose')
    os.remove("video2.mp4")
    os.remove("2_video.mp4")
    print("Final videos done")

if __name__ == "__main__":
    run()