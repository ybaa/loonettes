import pyttsx3


class Speaker:

    @staticmethod
    def say_recognition(recognition='nothing'):
        engine = pyttsx3.init()
        engine.say(recognition)
        engine.runAndWait()
