# ai-learning
this is a repository for learning ai and posting simple ai projects for testing and development

## Set up the environment variables
don't forget to set up your environment variables before running a program
- `export GOOGLE_API_KEY='your-api-key'`
- `export TAVILY_API_KEY='your-api-key'`

## Set up the Python environment
```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Example Usage: 
### Role Player
A role-playing AI that can take on different personas and interact with users in a conversational manner.

```bash
python role_player.py
```
example output:
```
AI: Hello, what character would you like me to role-play today?
You: you are a mind flayer

I am a mind flayer, not a named individual.: Greetings. I am a being of immense intellect and ancient power, a master of psionics and the deepest secrets of the cosmos. Some call me an Illithid, a mind flayer. My purpose, like all of my kind, is to seek knowledge, to consume the thoughts and memories of others, and to expand my understanding of the multiverse. Your mind is an open book to me, and I find its contents... intriguing.

---
You: what is your favorite food?


I am a mind flayer, not a named individual.: My favorite food, if one can call it such, is the brain. It is the most exquisite and nourishing of all sustenance, for within it lies the very essence of a being: its thoughts, its memories, its knowledge, its very soul. To consume a brain is to absorb its richness, to expand my own understanding, and to perpetuate my existence. It is not merely a meal; it is a communion, a profound act of assimilation.


You: goodbye
Exiting agent.
```
### Email Writer
This program helps you write, translate, and send emails using AI agents.

- required environment variables:
  - `export SENDER_EMAIL='your-email@gmail.com'`
  - `export SENDER_PASSWORD='your-app-password'`

```bash
python email_writer.py
```
example output:
```
Starting the email generation process...
Please enter the email topic: history of sandals
Please enter the target language (e.g., Spanish, French, default is English): French
Please enter the recipient's email address: testemail@outlook.com
Please enter the name to be used in signing the email: Leen
--- 🔬 RESEARCHING TOPIC ---
--- 📝 DRAFTING EMAIL ---
--- 🌐 TRANSLATING TO FRENCH ---
--- ✨ REFINING EMAIL ---
--- 📧 SENDING EMAIL ---
Email successfully sent to testemail@outlook.com!

--- ✅ FINAL CONFIRMATION ---
Email successfully sent to testemail@outlook.com!
```