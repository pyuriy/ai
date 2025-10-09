# Comprehensive Lab for Large Language Models (LLMs)

This hands-on lab guide builds on the LLM cheat sheet, providing practical exercises to explore LLMs from fundamentals to deployment. It's designed for beginners to intermediates, using free tools like Google Colab for accessibility. Exercises draw from established resources, with code snippets runnable in Colab (no local setup required unless noted). Each lab includes objectives, steps, code examples, and discussion questions. Aim for 2–4 hours per lab.

## Prerequisites
- **Environment**: Google Colab (free tier with T4 GPU access). No installation needed—notebooks run in-browser.
- **Skills**: Basic Python; familiarity with NumPy/Pandas (optional refresh in Lab 1).
- **Libraries**: Hugging Face Transformers, PyTorch, Datasets (auto-installed in Colab).
- **Accounts**: Free Hugging Face account for model downloads (optional for some labs).
- **Time**: 10–20 hours total for all labs.
- **Resources**: 
  - [Hands-On Large Language Models book notebooks](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) (12 chapters, visual & practical).
  - [LLM Course by Maxime Labonne](https://github.com/mlabonne/llm-course) (end-to-end, with 20+ Colab notebooks).

Start by forking the repos or opening linked Colabs.

## Lab 1: LLM Fundamentals & Setup
**Objectives**: Understand tokens/embeddings; set up environment.  
**Duration**: 1 hour.

### Steps
1. Open a new Colab notebook.
2. Install basics (run once):
   ```python:disable-run
   !pip install transformers torch datasets accelerate
   ```
3. Load a tokenizer and explore tokens:
   ```python
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   text = "Hello, world! How are you?"
   tokens = tokenizer.tokenize(text)
   print("Tokens:", tokens)
   print("Token IDs:", tokenizer.convert_tokens_to_ids(tokens))
   ```
   Expected output: Tokens like ['Hello', ',', ' world', '!'] with IDs [15496, 11, 995, 0, 1293, 509, 30, '?'].

4. Generate embeddings (vector reps):
   ```python
   import torch
   from transformers import AutoModel
   model = AutoModel.from_pretrained("gpt2")
   inputs = tokenizer(text, return_tensors="pt")
   outputs = model(**inputs)
   embeddings = outputs.last_hidden_state
   print("Embedding shape:", embeddings.shape)  # [1, num_tokens, 768]
   ```

**Discussion**: How do tokens affect model input size? Experiment with longer text—hit context limits?

## Lab 2: Prompt Engineering
**Objectives**: Master zero/few-shot, CoT prompting.  
**Duration**: 1.5 hours.  
*Based on Chapter 6 of Hands-On LLMs.*

### Steps
1. Load a generative model:
   ```python
   from transformers import pipeline
   generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
   ```

2. **Zero-Shot**: Direct task.
   ```python
   prompt = "Translate to French: Hello, how are you?"
   output = generator(prompt, max_length=50, num_return_sequences=1)
   print(output[0]['generated_text'])
   ```
   Expected: Something like "Translate to French: Hello, how are you? Bonjour, comment allez-vous?"

3. **Few-Shot**: Add examples.
   ```python
   few_shot = """Q: What is the capital of France? A: Paris.
   Q: What is the capital of Japan? A: Tokyo.
   Q: What is the capital of Brazil? A: """
   output = generator(few_shot, max_length=100)
   print(output[0]['generated_text'])
   ```

4. **Chain-of-Thought**: Step-by-step.
   ```python
   cot_prompt = "Solve: If a bat and ball cost $1.10, bat costs $1 more than ball. Ball cost? Think step by step."
   output = generator(cot_prompt, max_length=150)
   print(output[0]['generated_text'])
   ```

**Discussion**: Compare outputs—why does CoT reduce errors? Tweak temperature (e.g., `temperature=0.7`) for creativity.

## Lab 3: Text Classification & Embeddings
**Objectives**: Classify text; create custom embeddings.  
**Duration**: 2 hours.  
*Based on Chapters 4 & 10 of Hands-On LLMs.*

### Steps
1. Fine-tune for sentiment (use IMDB dataset):
   ```python
   from datasets import load_dataset
   from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
   dataset = load_dataset("imdb", split="train[:1000]")  # Small subset
   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   def tokenize(examples): return tokenizer(examples["text"], truncation=True)
   tokenized = dataset.map(tokenize, batched=True)
   model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
   training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=8)
   trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
   trainer.train()
   ```

2. Predict:
   ```python
   text = "This movie is amazing!"
   inputs = tokenizer(text, return_tensors="pt")
   outputs = model(**inputs)
   prediction = torch.argmax(outputs.logits, dim=1).item()
   print("Sentiment:", "Positive" if prediction == 1 else "Negative")
   ```

3. Embeddings for similarity:
   ```python
   from sentence_transformers import SentenceTransformer
   embedder = SentenceTransformer("all-MiniLM-L6-v2")
   sentences = ["I love AI", "LLMs are cool", "I hate bugs"]
   embeddings = embedder.encode(sentences)
   from sklearn.metrics.pairwise import cosine_similarity
   sim = cosine_similarity([embeddings[0]], embeddings)
   print("Similarities:", sim[0])
   ```

**Discussion**: How does fine-tuning improve accuracy? Visualize embeddings with t-SNE (add `!pip install scikit-learn`).

## Lab 4: Retrieval-Augmented Generation (RAG)
**Objectives**: Build a simple RAG pipeline.  
**Duration**: 2 hours.  
*Based on LLM Course Part 3 & Hands-On Chapter 8.*

### Steps
1. Install extras: `!pip install langchain chromadb sentence-transformers`
2. Ingest docs and query:
   ```python
   from langchain.document_loaders import TextLoader
   from langchain.text_splitter import CharacterTextSplitter
   from langchain.embeddings import HuggingFaceEmbeddings
   from langchain.vectorstores import Chroma
   from langchain.chains import RetrievalQA
   from langchain.llms import HuggingFacePipeline

   # Sample doc (replace with your text file)
   loader = TextLoader("state_of_the_union.txt")  # Download sample from datasets
   from datasets import load_dataset; ds = load_dataset("wikitext", "wikitext-2-raw-v1"); with open("doc.txt", "w") as f: f.write(ds["train"][0]["text"])
   loader = TextLoader("doc.txt")
   documents = loader.load()
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   texts = text_splitter.split_documents(documents)
   embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   db = Chroma.from_documents(texts, embeddings)
   llm = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")
   qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
   query = "What is the main topic?"
   print(qa.run(query))
   ```

**Discussion**: How does RAG reduce hallucinations? Test with irrelevant queries.

## Lab 5: Fine-Tuning a Model
**Objectives**: Fine-tune for custom tasks using LoRA.  
**Duration**: 3 hours (GPU recommended).  
*Based on LLM Course Fine-Tuning Notebooks & Hands-On Chapter 12.*

### Steps
1. Use Unsloth for efficient fine-tuning (Llama 3.1 example):
   ```python
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained("unsloth/llama-3.1-8b-bnb-4bit", dtype=None, load_in_4bit=True)
   model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_alpha=16, lora_dropout=0)
   # Add your dataset (e.g., Alpaca format)
   from datasets import load_dataset
   alpaca = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
   def formatting_prompts_func(examples):
       instructions = examples["instruction"]
       outputs = examples["output"]
       texts = []
       for instruction, output in zip(instructions, outputs):
           text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}" + tokenizer.eos_token
           texts.append(text)
       return {"text": texts}
   alpaca = alpaca.map(formatting_prompts_func, batched=True)
   from trl import SFTTrainer
   trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=alpaca, dataset_text_field="text", max_seq_length=512)
   trainer.train()
   ```

2. Save and test:
   ```python
   model.save_pretrained("lora_model")
   FastLanguageModel.for_inference(model)
   inputs = tokenizer("### Instruction:\nWhat is AI?\n\n### Response:\n", return_tensors="pt").to("cuda")
   outputs = model.generate(**inputs, max_new_tokens=64)
   print(tokenizer.decode(outputs[0]))
   ```

**Discussion**: Monitor loss—why LoRA over full fine-tuning? Experiment with epochs.

## Lab 6: Agents & Deployment
**Objectives**: Build an LLM agent; deploy a simple app.  
**Duration**: 2.5 hours.  
*Based on LLM Course Part 3 & Hands-On Chapter 7.*

### Steps
1. Simple ReAct agent with LangChain:
   ```python
   !pip install langchain langchain-community langchain-openai
   from langchain.agents import load_tools, initialize_agent, AgentType
   from langchain.llms import HuggingFaceHub  # Use free HF token
   llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5})
   tools = load_tools(["serpapi", "llm-math"], llm=llm)  # SerpAPI needs key; skip or mock
   agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
   agent.run("What is 2+2? Then search for latest LLM news.")
   ```

2. Deploy with Gradio:
   ```python
   !pip install gradio
   import gradio as gr
   def generate(text): return generator(text, max_length=100)[0]['generated_text']
   iface = gr.Interface(fn=generate, inputs="text", outputs="text")
   iface.launch(share=True)  # Public link for 72h
   ```

**Discussion**: How do agents handle multi-step tasks? Secure APIs in production.

## Lab 7: Evaluation & Optimization
**Objectives**: Evaluate models; quantize for efficiency.  
**Duration**: 2 hours.  
*Based on LLM Course Evaluation & Quantization.*

### Steps
1. Basic eval with ROUGE:
   ```python
   !pip install rouge-score
   from rouge_score import rouge_scorer
   scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
   scores = scorer.score("The cat sat on the mat.", "A feline rested on the rug.")
   print(scores)
   ```

2. Quantize a model:
   ```python
   !pip install bitsandbytes
   from transformers import BitsAndBytesConfig
   quantization_config = BitsAndBytesConfig(load_in_4bit=True)
   model = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=quantization_config, device_map="auto")
   print("Quantized model loaded!")
   ```

**Discussion**: Trade-offs of quantization (speed vs. accuracy)?

## Next Steps & Challenges
- **Advanced**: Try multimodal (Lab from Hands-On Ch. 9: CLIP for image-text).
- **Project**: Build a RAG chatbot for your domain.
- **Challenges**: Fine-tune on custom data; merge models with MergeKit.
- **Track Progress**: Use Weights & Biases (`!pip install wandb`) for logging.

This lab emphasizes experimentation—fork notebooks, iterate! For full code, see the cited repos.
```
