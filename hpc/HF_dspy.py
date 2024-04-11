import dspy


HF_TOKEN='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'
mistral = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.2',token=HF_TOKEN)

response = mistral("Tell me a joke")
print(response)