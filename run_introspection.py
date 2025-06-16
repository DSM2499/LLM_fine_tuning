from introspect import Introspector
import os
import json


sample_tasks = [
    {"prompt": "Summarize this article about climate change.", "response": "The weather is getting warmer."},
    {"prompt": "Explain how gradient descent works.", "response": "Gradient descent is like rolling down a hill."},
    {"prompt": "Write a haiku about the moon.", "response": "The moon glows softly / Night’s silver mirror shines down / Shadows dance in light."},
    {"prompt": "Translate this to French: 'Good morning!'", "response": "Bonjour!"},
    {"prompt": "What is the capital of Brazil?", "response": "Rio de Janeiro"}
]

introspector = Introspector()
critique = introspector.run(sample_tasks)

print("\n===== GPT-4 Critique and Suggestions =====\n")
print(critique)