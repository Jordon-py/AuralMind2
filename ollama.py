from ollama import Client

OLLAMA_API_KEY='cd818f2bdfd74ce8b0392665e87905f0.fb1fMJWwjMElmh7AwJGwHc1B'

client = Client(OLLAMA_API_KEY)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)

