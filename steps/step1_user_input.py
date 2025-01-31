import speech_recognition as sr

def get_user_input(input_type, video_id):
    """
    Captures user input and processes it.
    - If voice input is selected, it uses speech recognition for the entire session.
    - If text input is selected, it uses text input for the entire session.
    - The video ID entered at the beginning is used throughout the session.
    """
    if input_type == "1":  # Text input session
        print("Type '/bye' to end the session.")
        while True:
            try:
                user_query = input("You: ").strip()
                if user_query.lower() == "/bye":
                    print("Goodbye!")
                    break
                yield user_query, video_id
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break

    elif input_type == "2":  # Voice input session
        print("Say 'Goodbye' to end the session. Press Ctrl+C to interrupt.")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            while True:
                try:
                    print("Speak your query:")
                    audio = recognizer.listen(source)
                    user_query = recognizer.recognize_google(audio)
                    print(f"Recognized: {user_query}")
                    if user_query.lower() == "goodbye":
                        print("Goodbye!")
                        break
                    yield user_query, video_id
                    print("How may I assist you further?")
                except sr.UnknownValueError:
                    print("Could not understand the audio. Please try again.")
                except sr.RequestError as e:
                    print(f"Error with speech recognition service: {e}")
                except KeyboardInterrupt:
                    print("\nSession interrupted. Goodbye!")
                    break

    else:
        print("Invalid input type. Please restart and choose either (1) Text or (2) Voice.")