from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel


model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

prompt = "The future of AI in education is"
outputs = generator(prompt, max_new_tokens=100, num_return_sequences=1)


print("Generated Text:")
print(outputs[0]['generated_text'])
