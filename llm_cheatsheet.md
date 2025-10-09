# Comprehensive Cheat Sheet for Large Language Models (LLMs)

## 🧠 What is an LLM?
Large Language Models (LLMs) are advanced AI systems trained on massive datasets of natural language to process, understand, generate, and reason with human-like text. They form the backbone of generative AI, enabling tasks like chatbots, translation, and code generation. LLMs excel at pattern recognition but may hallucinate facts, so outputs require verification.

## 🔑 Core Concepts
- **Tokens**: Basic units LLMs process (words or subwords); e.g., "running" might split into "run" + "ning".
- **Embeddings**: Vector representations of tokens capturing semantic meaning; positional encoding adds sequence order.
- **Context Window**: Maximum tokens the model can "see" at once (e.g., 128K for GPT-4o).
- **Attention Mechanism**: Computes relationships between tokens, allowing focus on relevant parts regardless of position.
- **Autoregressive vs. Masked Modeling**: Autoregressive (e.g., GPT) predicts next token; masked (e.g., BERT) fills blanks in bidirectional context.
- **Temperature**: Controls output randomness (low = deterministic, high = creative).
- **Top-k/Top-p Sampling**: Limits token choices for diversity (k = fixed count, p = cumulative probability).

## 🏗️ Architectures
| Type              | Description                              | Examples              |
|-------------------|------------------------------------------|-----------------------|
| Transformer      | Core architecture with self-attention and feedforward layers. | GPT series, BERT     |
| Encoder-Decoder  | Separate encoders/decoders for input/output (e.g., translation). | T5, BART             |
| Decoder-Only     | Simplified for generation; causal masking. | GPT, LLaMA           |
| Mixture of Experts (MoE) | Routes inputs to specialized "expert" sub-networks for efficiency. | Mixtral, Grok-1      |
| Multimodal       | Handles text + images/audio/video.       | Gemini, CLIP         |

## 🎓 Training and Fine-Tuning
### Training Phases
1. **Pre-Training**: Self-supervised on vast corpora (e.g., internet text) to learn language patterns.
2. **Instruction Tuning**: Aligns model to follow user commands.
3. **RLHF/RLAIF**: Reinforcement Learning from Human/AI Feedback to improve helpfulness and reduce biases.

### Fine-Tuning Techniques
- **LoRA/QLoRA**: Low-Rank Adaptation for efficient updates without full retraining; quantized for lower memory.
- **RAG (Retrieval-Augmented Generation)**: Combines LLM with external knowledge retrieval to reduce hallucinations.
- **Costs**: Training from scratch is expensive (e.g., GPT-4: ~$78M); fine-tuning open models is more accessible.

### Data & Compute
- Requires terabytes of cleaned data, GPUs/TPUs, and expert teams.
- Open-source enables decentralized training, but closed models often lead in performance.

## 📝 Prompting Techniques
- **Zero-Shot**: Direct task without examples (e.g., "Translate to French: Hello").
- **Few-Shot**: Provide 1–5 examples in prompt for better accuracy.
- **Chain-of-Thought (CoT)**: Prompt step-by-step reasoning (e.g., "Think step by step before answering").
- **Role Prompting**: Assign persona (e.g., "You are a helpful teacher").
- **Tips**: Keep prompts clear/short; use system messages for instructions; iterate based on outputs.

## 📊 Evaluation Metrics & Benchmarks
| Metric/Benchmark | Description                              | Use Case              |
|------------------|------------------------------------------|-----------------------|
| Perplexity      | Measures predictive uncertainty (lower = better). | Language modeling    |
| BLEU/ROUGE      | Compares generated vs. reference text.   | Translation/Summarization |
| F1 Score        | Balances precision/recall for classification. | NER, Sentiment       |
| MMLU            | Multi-task benchmark for knowledge/reasoning. | General intelligence |
| HumanEval       | Code generation accuracy.                | Programming tasks    |
| GLUE            | Suite for NLU tasks.                     | Understanding        |
| GSM8K           | Grade-school math problems.              | Reasoning            |

- **Other Checks**: Factuality (hallucination rate), coherence, toxicity/bias, latency/cost.

## 🚀 Applications & Use Cases
- **Chatbots/Virtual Assistants**: Customer support, conversations.
- **Content Generation**: Writing, summarization, paraphrasing.
- **Code Generation/Debugging**: Autocomplete, bug fixes.
- **Sentiment Analysis & Classification**: Emotion detection, theme clustering.
- **Translation & Global Content**: Multi-language support.
- **Agents**: Multi-step tasks with tools (e.g., planning, memory).

## 🛠️ Tools & Libraries
| Category        | Tools/Libraries                          |
|-----------------|------------------------------------------|
| Frameworks     | PyTorch, TensorFlow, Hugging Face Transformers |
| Prompting      | LangChain, PromptLayer                   |
| Retrieval      | FAISS, ChromaDB, Weaviate                |
| Fine-Tuning    | PEFT, LoRA, QLoRA, FlashAttention        |
| Agents         | CrewAI, AutoGen, LangGraph               |
| Deployment     | FastAPI, Docker, Hugging Face Spaces     |
| Monitoring     | Weights & Biases, Helicone               |
| Other          | Guidance (structured output), LlamaIndex (RAG apps), Haystack (search/Q&A) |

## 🔥 Top Open-Source LLMs (2025 Edition)
| Model           | Developer | Key Features                     |
|-----------------|-----------|----------------------------------|
| LLaMA 3        | Meta     | State-of-the-art general-purpose |
| Mistral/Mixtral| Mistral  | Efficient MoE, open weights     |
| Gemma          | Google   | Lightweight, high efficiency    |
| Phi-3          | Microsoft| Tiny model, strong performance  |
| DeepSeek       | DeepSeek | Excellent for code/general      |
| Zephyr         | H2O.ai   | RLHF-aligned chatbot            |

## ⚙️ Optimization & Best Practices
- **Efficiency**: Use quantization, shorter contexts, caching.
- **Ethics**: Mitigate biases, ensure privacy in fine-tuning.
- **Roadmap**: Start with basics (Python/ML), master Transformers, practice prompting, build RAG/agents.
- **Future Trends**: Multimodal integration, agentic AI, smaller efficient models (SLMs).

This cheat sheet draws from multiple expert resources for a balanced, up-to-date overview. For deeper dives, explore the cited sources.
