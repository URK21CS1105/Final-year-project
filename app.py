import winsound
from gtts import gTTS
from pydub import AudioSegment
import os

# Add ffmpeg to PATH in your virtual environment session
os.environ["PATH"] += os.pathsep + r"C:\Users\hp\Downloads\ffmpeg-20250115T091316Z-001\ffmpeg\bin"  # Replace with the actual path to ffmpeg

# def tts(text, lang='en', slow=False):
#     # tts_t = gTTS(text=text, lang=lang, slow=slow)
#     tts_t = gTTS(text)
#     tts_t.save("sounds/audio.mp3")
#     sound = AudioSegment.from_mp3("sounds/audio.mp3")
#     sound.export("sounds/audio.wav", format="wav")
#     winsound.PlaySound("sounds/audio.wav",winsound.SND_NODEFAULT |  winsound.SND_FILENAME)

# tts("Hello, There is a person ahead")

import winsound
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import tempfile
import os

def tts(text,):
    # Text to Speech conversion using gTTS
    tts_t = gTTS(text,slow=False)
    
    # Save to an in-memory bytes buffer
    mp3_fp = io.BytesIO()
    tts_t.write_to_fp(mp3_fp)
    mp3_fp.seek(0)  # Seek to the beginning of the in-memory buffer
    
    # Load the audio using pydub
    sound = AudioSegment.from_file(mp3_fp, format="mp3")
    
    # Create a custom temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="C:/Users/hp/Downloads/Blind-20250115T085514Z-001/Blind/blind") as tmp_file:
        temp_wav_path = tmp_file.name
        sound.export(temp_wav_path, format="wav")
    
    # Play the sound using winsound
    winsound.PlaySound(temp_wav_path, winsound.SND_FILENAME)
    
    # Clean up the temporary file after playing the sound
    os.remove(temp_wav_path)

# Example usage
# tts("Hello, There is a person ahead")
