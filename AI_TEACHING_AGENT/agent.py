

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
