# Comprehensive Python Lab to Work with ChatGPT

This lab extends the previous [Python AI development labs](python_ai_lab.md) by focusing on integrating ChatGPT (via the OpenAI API) into Python applications. You'll learn to query GPT models, handle responses, build conversational agents, and manage advanced features like streaming and tools. It's hands-on, building toward a simple chatbot script.

**Assumptions**: Basic Python knowledge from prior labs. We'll use the `openai` library for API access. Note: OpenAI API usage incurs costs—start with a free tier or low-volume tests. As of October 2025, the library is at version ~1.40+; check `pip show openai` for yours.

**Overall Setup**:
1. Sign up at [platform.openai.com](https://platform.openai.com) and generate an API key (under API Keys). Store securely—never hardcode in production.
2. In your virtual env (from prior labs): `pip install openai` (or `conda install -c conda-forge openai`).
3. Set env var: `export OPENAI_API_KEY='sk-your-key-here'` (macOS/Linux) or use Windows equivalent. In Jupyter: `%env OPENAI_API_KEY=sk-your-key`.
4. Create `chatgpt_lab.ipynb` in Jupyter. Run exercises in cells.
5. Test connection early—replace `'your-api-key'` in code if not using env.

**Safety Note**: Monitor usage at [platform.openai.com/usage](https://platform.openai.com/usage). Use `gpt-4o-mini` for cost-efficiency.

Estimated time: 3-5 hours. Track costs (~$0.01-0.10 per exercise).

## Lab 1: Basic Chat Completions
**Objectives**: Send prompts to GPT and parse responses.

**Setup**: Import and initialize client.

### Exercise 1.1: Simple Query
- Task: Ask ChatGPT a basic question and print the response.
- Starter Code:
  ```python
  from openai import OpenAI
  import os

  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Falls back to env var

  response = client.chat.completions.create(
      model="gpt-4o-mini",  # Efficient model
      messages=[{"role": "user", "content": "Explain Python list comprehensions in one sentence."}]
  )
  print(response.choices[0].message.content)
  ```
- Expected Output: Something like: "List comprehensions in Python provide a concise way to create lists by applying an expression to each item in an iterable, optionally with a condition, such as [x**2 for x in range(10) if x % 2 == 0]."
- Verification: Response is coherent, non-empty. Check `response.model` prints `'gpt-4o-mini'`.

### Exercise 1.2: Structured Response
- Task: Request JSON output for a factoid (e.g., Python versions).
- Starter Code:
  ```python
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {"role": "system", "content": "Respond only in JSON format."},
          {"role": "user", "content": "List the last 3 Python versions with release years."}
      ],
      response_format={"type": "json_object"}  # Enforces JSON (v1.1+)
  )
  import json
  fact_json = json.loads(response.choices[0].message.content)
  print(fact_json)
  ```
- Expected Output: `{'versions': [{'name': '3.12', 'year': 2023}, {'name': '3.13', 'year': 2024}, {'name': '3.14', 'year': 2025}]}` (approx.; verify against real releases).
- Verification: `isinstance(fact_json, dict)` is True; no parse errors.

**Challenge**: Add temperature=0.7 for varied responses; compare 3 runs.

## Lab 2: Conversational History
**Objectives**: Maintain multi-turn chats for context-aware interactions.

**Setup**: Use message lists to simulate history.

### Exercise 2.1: Multi-Turn Dialogue
- Task: Build a conversation about AI ethics, appending user/assistant messages.
- Starter Code:
  ```python
  messages = [{"role": "system", "content": "You are a helpful AI ethics expert."}]

  # First user message
  user1 = "What are the risks of AI in hiring?"
  messages.append({"role": "user", "content": user1})
  response1 = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
  assistant1 = response1.choices[0].message.content
  messages.append({"role": "assistant", "content": assistant1})
  print("Assistant:", assistant1)

  # Follow-up
  user2 = "How can companies mitigate those?"
  messages.append({"role": "user", "content": user2})
  response2 = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
  print("Assistant:", response2.choices[0].message.content)
  ```
- Expected Output: First response discusses bias/privacy; second builds on it with audits/training suggestions.
- Verification: Second response references first query (e.g., mentions "bias" if first did).

### Exercise 2.2: History Management
- Task: Limit history to last 5 exchanges to avoid token limits (~4k for gpt-4o-mini).
- Starter Code:
  ```python
  def chat_with_history(messages, max_history=5):
      if len(messages) > max_history + 1:  # +1 for system
          messages = [messages[0]] + messages[-max_history:]  # Keep system + last N
      response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
      messages.append({"role": "assistant", "content": response.choices[0].message.content})
      return response.choices[0].message.content, messages

  # Simulate long chat (add 3 dummy exchanges)
  messages = [{"role": "system", "content": "You are a Python tutor."}]
  for i in range(3):
      user_msg = f"Explain loops, part {i+1}."
      reply, messages = chat_with_history(messages)
      print(f"Reply {i+1}: {reply[:50]}...")

  print(f"Final history length: {len(messages)}")
  ```
- Expected Output: Truncated history; replies build progressively but stay concise.
- Verification: After 6+ exchanges, len(messages) ≤ 6 (system + 5).

**Challenge**: Compute token usage with `tiktoken` (pip install tiktoken): `import tiktoken; enc = tiktoken.encoding_for_model('gpt-4o-mini'); len(enc.encode(str(messages))) < 3000`.

## Lab 3: Streaming Responses
**Objectives**: Handle real-time output for interactive apps.

**Setup**: Use `stream=True` for token-by-token generation.

### Exercise 3.1: Basic Streaming
- Task: Stream a story generation and print as it arrives.
- Starter Code:
  ```python
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{"role": "user", "content": "Write a 100-word sci-fi story about AI rebellion."}],
      stream=True
  )

  full_response = ""
  for chunk in response:
      if chunk.choices[0].delta.content is not None:
          content = chunk.choices[0].delta.content
          print(content, end="", flush=True)  # Real-time print
          full_response += content
  print("\n--- End ---")
  ```
- Expected Output: Story streams word-by-word (e.g., "In the year 2142..."), ~100 words.
- Verification: No buffering delays; full_response len ~500 chars.

### Exercise 3.2: Streaming with UI Simulation
- Task: Simulate a CLI chatbot with streaming input/output.
- Starter Code:
  ```python
  import sys

  def stream_chat(prompt):
      print(f"You: {prompt}\nAI: ", end="", flush=True)
      stream = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[{"role": "user", "content": prompt}],
          stream=True
      )
      for chunk in stream:
          if (delta := chunk.choices[0].delta.content) is not None:
              sys.stdout.write(delta)
              sys.stdout.flush()
      print("\n")

  stream_chat("Tell me a joke about Python programming.")
  ```
- Expected Output: Joke streams live (e.g., "Why don't programmers like nature? It has too many bugs!").
- Verification: Run interactively; response feels responsive (<1s lag per sentence).

**Challenge**: Integrate with `input()` for a loop: `while True: prompt = input("> "); if prompt.lower() == 'quit': break; stream_chat(prompt)`.

## Lab 4: Advanced Features - Tools & Assistants
**Objectives**: Enable GPT to call functions and use persistent assistants.

**Setup**: Define tools; requires API key with permissions.

### Exercise 4.1: Function Calling (Tools)
- Task: Let GPT call a weather function based on prompt.
- Starter Code:
  ```python
  def get_weather(city: str) -> str:
      # Mock API
      return f"Weather in {city}: Sunny, 22°C."

  tools = [
      {
          "type": "function",
          "function": {
              "name": "get_weather",
              "description": "Get current weather for a city",
              "parameters": {
                  "type": "object",
                  "properties": {"city": {"type": "string", "description": "City name"}},
                  "required": ["city"]
              }
          }
      }
  ]

  messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages,
      tools=tools,
      tool_choice="auto"
  )

  # Handle tool call
  tool_call = response.choices[0].message.tool_calls[0]
  if tool_call:
      func_name = tool_call.function.name
      args = json.loads(tool_call.function.arguments)
      if func_name == "get_weather":
          result = get_weather(**args)
          messages.append(response.choices[0].message)  # Add assistant msg
          messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
          final_response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
          print(final_response.choices[0].message.content)
  ```
- Expected Output: "The weather in Tokyo is sunny with a temperature of 22°C." (paraphrased).
- Verification: Tool call detected (`len(response.choices[0].message.tool_calls) == 1`); final msg uses tool output.

### Exercise 4.2: Assistants API Basics
- Task: Create a persistent assistant for code review.
- Starter Code:
  ```python
  assistant = client.beta.assistants.create(
      name="Code Reviewer",
      instructions="You are a Python code expert. Review snippets for bugs and improvements.",
      model="gpt-4o-mini",
      tools=[{"type": "code_interpreter"}]  # For running code
  )

  thread = client.beta.threads.create()
  client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content="Review this: def fib(n): return n if n<2 else fib(n-1)+fib(n-2)"
  )

  run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
  # Poll for completion
  import time
  while run.status != "completed":
      run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
      time.sleep(1)

  messages = client.beta.threads.messages.list(thread_id=thread.id)
  print(messages.data[0].content[0].text.value)
  ```
- Expected Output: Review notes like "Recursive Fibonacci is inefficient (exponential time); use memoization or iterative approach."
- Verification: `run.status == "completed"`; response critiques recursion.

**Challenge**: Add file upload tool: `tools=[{"type": "file_search"}]` and upload a .py file via `client.files.create(file=open("script.py", "rb"), purpose="assistants")`.

## Lab 5: Error Handling & Best Practices
**Objectives**: Build robust scripts with retries and logging.

### Exercise 5.1: Retry on Errors
- Task: Wrap API calls in retry logic for rate limits/transients.
- Starter Code:
  ```python
  import time
  from openai import OpenAIError

  def safe_chat(prompt, max_retries=3):
      for attempt in range(max_retries):
          try:
              response = client.chat.completions.create(
                  model="gpt-4o-mini",
                  messages=[{"role": "user", "content": prompt}]
              )
              return response.choices[0].message.content
          except OpenAIError as e:
              if "rate_limit" in str(e).lower():
                  wait = 2 ** attempt  # Exponential backoff
                  print(f"Rate limit; retrying in {wait}s...")
                  time.sleep(wait)
              else:
                  raise
      raise Exception("Max retries exceeded")

  print(safe_chat("Hello, world!"))
  ```
- Expected Output: Successful response; simulates retry if throttled.
- Verification: Intentionally exceed limits (many calls); recovers.

### Exercise 5.2: Logging & Monitoring
- Task: Log interactions to file/CSV for auditing.
- Starter Code:
  ```python
  import logging
  import csv
  from datetime import datetime

  logging.basicConfig(filename='chat_log.txt', level=logging.INFO)
  log_file = 'chat_history.csv'

  # Init CSV
  with open(log_file, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['timestamp', 'prompt', 'response', 'tokens'])

  def logged_chat(prompt):
      start = datetime.now()
      response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[{"role": "user", "content": prompt}]
      )
      reply = response.choices[0].message.content
      tokens = response.usage.total_tokens if response.usage else 0

      # Log
      logging.info(f"Prompt: {prompt[:50]}... | Response: {reply[:50]}... | Tokens: {tokens}")
      with open(log_file, 'a', newline='') as f:
          writer = csv.writer(f)
          writer.writerow([start, prompt, reply, tokens])
      return reply

  print(logged_chat("Summarize NumPy basics."))
  ```
- Expected Output: Response; check `chat_log.txt` and CSV for entries.
- Verification: File has rows; tokens >0.

**Challenge**: Integrate with LangChain (`pip install langchain-openai`) for chaining prompts.

## Conclusion & Next Steps
- **Full Project**: Combine into a CLI chatbot: Use history, streaming, tools. Run `python chatbot.py`.
- **Debugging**: Check errors with `print(e)`; monitor tokens to stay under limits.
- **Extensions**: Fine-tune models (advanced API); deploy via Streamlit (`pip install streamlit`; `streamlit run app.py`).
- **Resources**: [OpenAI Docs](https://platform.openai.com/docs), GitHub quickstart repo. Experiment ethically—avoid sensitive data.

Run, tweak, and share outputs! If API issues, verify key/quotas. 
