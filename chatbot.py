import json
import random
import re
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load chatbot data
# -----------------------------

with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# -----------------------------
# Prepare training data
# -----------------------------

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])


vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(patterns)


# -----------------------------
# Conversation history
# -----------------------------

HISTORY_FILE = "data/chat_history.json"


def save_history(user_message, bot_message):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    conversation = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_message,
        "bot": bot_message
    }

    history.append(conversation)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


# -----------------------------
# Chatbot response
# -----------------------------

def get_response(user_input):

    user_vector = vectorizer.transform([user_input])

    similarities = cosine_similarity(
        user_vector,
        pattern_vectors
    )[0]

    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    if best_score < 0.25:
        return (
            "Sorry, I don't understand that yet. "
            "Try asking about Python, DBMS, studying or calculations."
        )

    matched_tag = tags[best_match_index]

    for intent in data["intents"]:

        if intent["tag"] == matched_tag:
            return random.choice(intent["responses"])


# -----------------------------
# Calculator
# -----------------------------

def calculate(expression):

    expression = expression.replace(" ", "")

    if not re.fullmatch(r"[0-9+\-*/().]+", expression):
        return "I can only calculate basic mathematical expressions."

    try:
        result = eval(
            expression,
            {"__builtins__": None},
            {}
        )

        return f"The answer is {result}"

    except:
        return "Sorry, I couldn't calculate that."


# -----------------------------
# Show conversation history
# -----------------------------

def show_history():

    try:

        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)

        if not history:
            print("\nNo conversation history found.\n")
            return

        print("\n========== CHAT HISTORY ==========")

        for item in history[-10:]:

            print("\nTime:", item["time"])
            print("You :", item["user"])
            print("Bot :", item["bot"])

        print("\n=================================\n")

    except (FileNotFoundError, json.JSONDecodeError):

        print("\nNo conversation history found.\n")


# -----------------------------
# Main chatbot
# -----------------------------

def chatbot():

    print("=" * 55)
    print("             STUDENT ASSISTANT")
    print("=" * 55)

    print("Commands:")
    print("  history  - Show previous conversations")
    print("  exit     - Close chatbot")
    print()

    while True:

        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Exit
        if user_input.lower() in ["exit", "quit", "bye"]:

            bot_response = "Goodbye! Keep learning!"

            print("Bot:", bot_response)

            save_history(user_input, bot_response)

            break

        # History
        if user_input.lower() == "history":

            show_history()
            continue

        # Calculator
        if user_input.lower().startswith("calculate "):

            expression = user_input[10:]

            bot_response = calculate(expression)

            print("Bot:", bot_response)

            save_history(user_input, bot_response)

            continue

        # Normal chatbot
        bot_response = get_response(user_input)

        print("Bot:", bot_response)

        save_history(user_input, bot_response)


# -----------------------------
# Start program
# -----------------------------

if __name__ == "__main__":
    chatbot()