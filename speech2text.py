# this code works!!!
import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

'''
# create a Microphone instance for the first working mic; mics must be ON to
# be recognized as working, and the first mic will be used
for device_index in sr.Microphone.list_working_microphones():
    print(device_index)
    #m = sr.Microphone(device_index=device_index)
    break
else:
    print("No working microphones found!")
    raise SystemExit
'''

# Reading Microphone as source
# listening the speech and store in audio variable
device_index = 0
with sr.Microphone(device_index=device_index) as source:
    # listen for x secs to calibrate energy threshold for ambient noise levels
    # comment: in my setup, I said "red toyota camry 3000 miles 4000 dollars"
    # for the following durations in seconds. Seems like
    # "adjust_for_ambient_noise(..) is useless in my setup:
    # 0 => red Toyota Camry 3000 me 4,000
    # 1 => most of the time I get
    #           "Google Speech Recognition could not understand audio"
    # 2 => red Toyota Camry 3000 me 4,000
    # 3 => red Toyota Camry 3000 me 4,000
    r.adjust_for_ambient_noise(source=source, duration=3)
    print("Talk")
    # allow user "timeout" secs to start speaking; allow user upto
    # "phrase_time_limit" secs to speak
    audio = r.listen(source=source, timeout=5.0, phrase_time_limit=15.0)
    print("Time over, thanks")

    # recoginze_() method will throw a request
    # error if the API is unreachable,
    # hence using exception handling
    try:
        # using google speech recognition
        print("Text: "+r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
