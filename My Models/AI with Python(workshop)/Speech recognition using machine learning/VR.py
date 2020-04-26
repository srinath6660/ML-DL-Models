import speech_recognition as sr
import subprocess



r = sr.Recognizer()
mic = sr.Microphone()
while true:
with mic as source :
    
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    transcript = r.recognize_google(audio)
    print(transcript)
    subprocess.call(['say','Hello! how are you?'])

    