import os
import wave
import pyaudio
import speech_recognition as sr
from dotenv import load_dotenv
from pydub import AudioSegment
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

audio_dir = "audio"
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

def record_audio_and_save(duration=5):
    """
    Records audio from the microphone for a specified duration and saves it to a file.
    The filename is a sequential number (e.g., 1.wav, 2.wav).
    """
    # Audio settings
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    rate = 16000

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open a stream
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    print(f"Recording for {duration} seconds...")
    frames = []
    
    # Record audio data
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Recording finished!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Determine the next sequential filename
    existing_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not existing_files:
        next_number = 1
    else:
        file_numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
        if file_numbers:
            next_number = max(file_numbers) + 1
        else:
            next_number = 1
            
    filename = os.path.join(audio_dir, f"{next_number}.wav")

    # Save the recorded audio to a file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        
    print(f"Audio saved as '{filename}'")
    return filename

def transcribe_audio_and_get_llm_response(audio_file_path):
    """
    Transcribes the audio file and sends the transcription to the LLM for a response.
    """
    recognizer = sr.Recognizer()
    
    # Transcribe the audio
    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_file_path)
        # Export as a temporary WAV file if it's not already
        if not audio_file_path.endswith('.wav'):
            temp_wav_path = "temp_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_file_path = temp_wav_path

        with sr.AudioFile(audio_file_path) as source:
            print("Transcribing audio...")
            audio_data = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio_data)
            print("Transcription successful!")
            
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return
    except Exception as e:
        print(f"An error occurred during audio processing: {e}")
        return
    
    print(f"\nTranscribed Text: {transcribed_text}")
    
    # Get response from the LLM
    try:
        response_prompt = f"Based on this transcription: '{transcribed_text}', please answer the question or provide a summary."
        print("\nSending transcription to LLM...")
        llm_response = llm.invoke(response_prompt)
        
        print("\nLLM's Response:")
        print(llm_response.content)
    except Exception as e:
        print(f"An error occurred with the LLM API: {e}")

# --- Main execution loop ---
if __name__ == "__main__":
    while True:
        try:
            # Record a new audio file
            recorded_file = record_audio_and_save(duration=6)
            
            # Process the recorded file
            transcribe_audio_and_get_llm_response(recorded_file)
            
            # Ask the user if they want to record again
            choice = input("\nDo you want to record another query? (yes/no): ").lower()
            if choice != 'yes':
                break
                
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break