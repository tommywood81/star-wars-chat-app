import requests
import json

# Get characters from API
response = requests.get("http://localhost:8002/characters")
data = response.json()

# Sort by dialogue count
characters = data['characters']
sorted_chars = sorted(characters, key=lambda x: x.get('dialogue_count', 0), reverse=True)

print("Top 15 characters by line count:")
for i, char in enumerate(sorted_chars[:15], 1):
    name = char['name']
    count = char.get('dialogue_count', 0)
    print(f"{i}. {name}: {count} lines")

# Custom quick start suggestions for top characters
character_suggestions = {
    "Luke Skywalker": [
        "Tell me about your father and the dark side",
        "What was it like training with Yoda on Dagobah?",
        "How did you feel when you found out about Leia?"
    ],
    "Han Solo": [
        "What's your honest opinion about the Force?",
        "Tell me about that time you shot first",
        "How do you really feel about Princess Leia?"
    ],
    "C-3PO": [
        "What's your protocol for dealing with droids?",
        "Tell me about your relationship with R2-D2",
        "What's the most dangerous situation you've been in?"
    ],
    "Leia": [
        "What's it like being a princess and a rebel?",
        "How do you handle being called 'Your Worship'?",
        "What's your strategy for dealing with the Empire?"
    ],
    "VADER": [
        "What do you think about your son's potential?",
        "How do you feel about the Emperor's plans?",
        "What's your opinion on the Death Star's effectiveness?"
    ],
    "Obi-Wan": [
        "What was it like training Anakin Skywalker?",
        "How do you feel about becoming a Force ghost?",
        "What's your advice for young Jedi?"
    ],
    "Yoda": [
        "What's the most important lesson you teach?",
        "How do you feel about Luke's training?",
        "What's your philosophy on fear and anger?"
    ],
    "Chewbacca": [
        "What's your favorite thing about the Millennium Falcon?",
        "How do you communicate with Han?",
        "What's your opinion on human pilots?"
    ],
    "R2-D2": [
        "What's your most impressive technical achievement?",
        "How do you feel about C-3PO's constant worrying?",
        "What's your secret to surviving so many battles?"
    ],
    "Lando": [
        "What was it like running Cloud City?",
        "How do you feel about your deal with Vader?",
        "What's your strategy for winning at sabacc?"
    ],
    "Emperor": [
        "What's your philosophy on power and control?",
        "How do you feel about the Rebel Alliance?",
        "What's your opinion on the Force's dark side?"
    ],
    "Boba Fett": [
        "What's your approach to bounty hunting?",
        "How do you feel about working for the Empire?",
        "What's your most challenging target?"
    ],
    "Admiral Piett": [
        "What's it like serving under Darth Vader?",
        "How do you handle the pressure of command?",
        "What's your strategy for dealing with the Rebels?"
    ],
    "Admiral Ackbar": [
        "What's your military strategy against the Empire?",
        "How do you feel about the Rebel Alliance?",
        "What's your opinion on the Death Star II?"
    ],
    "Wedge": [
        "What's it like being a Red Squadron pilot?",
        "How do you feel about the Death Star trench run?",
        "What's your advice for new pilots?"
    ]
}

print("\nCustom suggestions for each character:")
for char in sorted_chars[:15]:
    name = char['name']
    if name in character_suggestions:
        print(f"\n{name}:")
        for suggestion in character_suggestions[name]:
            print(f"  - {suggestion}")
