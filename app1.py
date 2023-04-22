import openai
import gradio as gr
from pydub import AudioSegment
import config
from gtts import gTTS
from playsound import playsound
import os

#Initiating openai API key
openai.api_key = config.api_key

# List of all the conversation log between PGT and the user
chat_records = [{
            "role": "system", 
            "content": "You are a medical doctor. Answer each question in 40 words or less. Add at the end 'This is not a diagnosis'. Simplify the answer and explain any clinical terms.'"
            }, ]

# Text to speech function 
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False, tld='us')
    tts.save("response.mp3")
    playsound("response.mp3")
    os.remove("response.mp3")

def transcribe(audio):
    global chat_records
    print("Received audio file:", audio)
   
    if audio is None:
        return "Error: No audio input received. Please try again."
    
    # Convert audio to 16-bit format
    audio_segment = AudioSegment.from_file(audio)
    audio_segment = audio_segment.set_sample_width(2)
    audio_segment.export("converted_audio.wav", format="wav")
    
    # Transcribing the file using openAI whisper API
    audio_file = open("converted_audio.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
    
    # Saving the user conversation to the list
    user_message = transcript["text"]
    chat_records.append({"role": "user", "content": user_message})

    # Creating system response using openAI ChatCompleteion API
    gpt_response = openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
         messages=chat_records
    )
    print(gpt_response)

    # Saving GPT response into the list and talking back to the user
    gpt_message = gpt_response["choices"][0]["message"]["content"]
    print(gpt_message)
    chat_records.append({"role":"assistant", "content":gpt_message})

    # Run text_to_speech2 
    #tts_thread = threading.Thread(target=text_to_speech, args=(gpt_message,))
    #tts_thread.start()
    text_to_speech(gpt_message)

    # Cleaning the output (chat_records) to show on the screen
    full_chat =""
    for message in chat_records:
        if message['role'] != 'system': # Removing the set up prompt
            full_chat += message['role'] +" : " + message['content'] + "\n\n"

    return full_chat

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text").launch(share=True)
ui.launch()