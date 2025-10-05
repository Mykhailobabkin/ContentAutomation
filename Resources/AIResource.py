from openai import OpenAI
import os
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types
import mimetypes
import wave
import base64

load_dotenv()

class OpenAIRequest:
    """
    Handles requests to the OpenAI API.
    """

    def __init__(self, model: str):
        """
        Initialize the OpenAIRequest with a model name.
        Loads the API key from environment variables.
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def send_request(self, prompt: str, promt1: str = None, promt2: str = None, promt3: str = None):
        """
        Sends a prompt to the OpenAI API with optional additional messages.
        
        Args:
            prompt (str): Main prompt.
            promt1, promt2, promt3 (str, optional): Additional prompts.
        
        Returns:
            str: The output text from the response, or None if an error occurs.
        """
        try:
            messages = [{"role": "developer", "content": prompt}]
            if promt1:
                messages.append({"role": "user", "content": promt1})
            if promt2:
                messages.append({"role": "user", "content": promt2})
            if promt3:
                messages.append({"role": "user", "content": promt3})
            response = self.client.responses.create(
                model=self.model,
                input=messages
            )
            return response.output_text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def internet_search(self, prompt: str):
        """
        Performs an internet search using the OpenAI API's web search tool.
        
        Args:
            prompt (str): Search query.
        
        Returns:
            str: The output text from the response, or None if an error occurs.
        """
        try: 
            response = self.client.response.create(
                model=self.model,
                tools=[{
                    "type": "web_search",
                    "search_context_size": "high",
                }],
                input=prompt
            )
            return response.output_text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

class GeminiRequest:
    """
    Handles requests to the Google Gemini API.
    """

    def __init__(self, model: str):
        """
        Initialize the GeminiRequest with a model name.
        Loads the API key from environment variables.
        """
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def send_request(self, prompt: str):
        """
        Sends a prompt to the Gemini API.
        
        Args:
            prompt (str): The prompt to send.
        
        Returns:
            str: The response text, or None if an error occurs.
        """
        try: 
            response = self.client.models.generate_content(
                model=self.model, 
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def internet_search(self, prompt: str):
        """
        Performs an internet search using Gemini's Google Search tool.
        
        Args:
            prompt (str): Search query.
        
        Returns:
            str: The response text, or None if an error occurs.
        """
        try: 
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            config = types.GenerateContentConfig(
                tools=[grounding_tool],
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def analyze_video(self, video_file_path, prompt: str):
        """
        Analyzes a video file using Gemini API.
        
        Args:
            video_file_path (str): Path to the video file.
            prompt (str): Prompt for analysis.
        
        Returns:
            str: The response text, or error message if failed.
        """
        try:
            mime_type = self.get_mime_type(video_file_path)
            with open(video_file_path, "rb") as video_file:
                video_bytes = video_file.read()
            response = self.client.models.generate_content(
                model=self.model,
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type=mime_type)
                        ),
                        types.Part(text=prompt)
                    ] 
                )
            )
            return response.text
        except Exception as e:
            print(f"Video analysis failed: {e}")
            return f"An error occurred during video analysis: {e}"

    def get_mime_type(self, file_path):
        """
        Determines the MIME type of a file based on its extension.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            str: MIME type.
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type
        file_ext = os.path.splitext(file_path)[1].lower()
        mime_mapping = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.m4v': 'video/mp4',
            '.3gp': 'video/3gpp',
            '.flv': 'video/x-flv'
        }
        return mime_mapping.get(file_ext, 'video/mp4')

    def analyze_yt_video(self, video_link, prompt: str):
        """
        Analyzes a YouTube video using Gemini API.
        
        Args:
            video_link (str): URL to the YouTube video.
            prompt (str): Prompt for analysis.
        
        Returns:
            str: The response text.
        """
        response = self.client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=video_link)
                    ),
                    types.Part(text=prompt)
                ]
            )
        )
        return response.text

    def generate_video(self, prompt: str):
        """
        Generates a video using Gemini API.
        
        Args:
            prompt (str): Prompt for video generation.
        
        Returns:
            None if an error occurs.
        """
        try: 
            operation = self.client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio="9:16",  
                    person_generation="allow_all"
                )
            )
            while not operation.done:
                print("Waiting for video generation to complete...")
                time.sleep(10)
                operation = self.client.operations.get(operation)
            video = operation.response.generated_videos[0]
            self.client.files.download(file=video.video)
            video.video.save("video.mp4")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def generate_video_w_images(self, prompt: str, image_path, output_path: str):
        """
        Generates a video using Gemini API with an input image.
        
        Args:
            prompt (str): Prompt for video generation.
            image_path (str): Path to the input image.
        
        Returns:
            None if an error occurs.
        """
        try:
            # Read the image file as bytes
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Encode to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine MIME type based on file extension
            mime_type = 'image/jpeg'
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith('.webp'):
                mime_type = 'image/webp'
            
            # Create the image dictionary with the correct structure
            image = {
                'imageBytes': image_base64,
                'mimeType': mime_type
            }
            
            operation = self.client.models.generate_videos(
                model=self.model,
                prompt=prompt,
                image=image,
                config=types.GenerateVideosConfig(
                    aspect_ratio="9:16",  
                    person_generation="allow_adult"
                )
            )
            
            while not operation.done:
                print("Waiting for video generation to complete...")
                time.sleep(10)
                operation = self.client.operations.get(operation)
            
            video = operation.response.generated_videos[0]
            self.client.files.download(file=video.video)
            video.video.save(output_path)
            
            print("Video generated successfully!")
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def wave_file(self, filename, pcm, channels=1, rate=24000, sample_width=2):
        """
        Saves PCM audio data to a WAV file.
        
        Args:
            filename (str): Output WAV file name.
            pcm (bytes): PCM audio data.
            channels (int): Number of audio channels.
            rate (int): Sample rate.
            sample_width (int): Sample width in bytes.
        """
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)

    def generate_audio(self, prompt: str):
        """
        Generates audio from text using Gemini API.
        
        Args:
            prompt (str): Prompt for audio generation.
        
        Returns:
            None
        """
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',
                        )
                    )
                ),
            )
        )
        data = response.candidates[0].content.parts[0].inline_data.data
        file_name = 'out.wav'
        self.wave_file(file_name, data)