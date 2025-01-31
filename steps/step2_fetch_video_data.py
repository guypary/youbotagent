from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from pytube import YouTube
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def fetch_video_data(video_id):
    """
    Fetches metadata and transcript (either user-uploaded or auto-generated) for a given YouTube video ID.
    """
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    # Fetch video metadata
    video_response = youtube.videos().list(
        part="snippet,contentDetails",
        id=video_id
    ).execute()

    if video_response['items']:
        video_data = video_response['items'][0]['snippet']
        print(f"Video Title: {video_data['title']}")
        #print(f"Description: {video_data['description']}")

        # Fetch transcript using youtube-transcript-api
        try:
            print("Fetching transcript...")
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'a.en'])
            # Combine the transcript into a single string
            transcript_text = " ".join([item['text'] for item in transcript])
            return transcript_text

        except TranscriptsDisabled:
            print("Transcripts are disabled for this video.")
            return None
        except NoTranscriptFound:
            print("No transcript (user-uploaded or auto-generated) found for this video.")
            return None
        except VideoUnavailable:
            print("This video is unavailable.")
            return None
        except Exception as e:
            print(f"An error occurred while fetching the transcript: {e}")
            return None
    else:
        print("Video not found.")
        return None