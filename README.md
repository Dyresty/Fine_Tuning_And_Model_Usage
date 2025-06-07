# Fine Tuning and Model Usage

Fine tuning of a causal AI, Usage of Grok api and running of Mistral 7B model locally.

### Fine Tuning a Causal Language Model with Hugging Face Transformers
#### Causal Language Model
Causal Language Model (CLM) is a type of language model that generates text by predicting the next word/token in a sequence, based only on the tokens before ot. 

Predicts 'what comes next', so used ideally in text generation tasks. Model sees text from left to right and cannot peek ahead. 
Example - GPT, DialoGPT, LLaMa, Falcon. 

If the sentence is:
"The cat sat on the"

A causal model tries to guess the next word, like:
"The cat sat on the mat"
But only having seen: The, cat, sat, on, the.

#### Hugging Face Libraries 
1. Transformers
The core library for using and fine-tuning pretrained models.
- Provides thousands of pretrained models for NLP, vision, audio, and multimodal tasks.
- Unified API for tokenization, model loading, training, generation, etc.
2. Datasets
For loading, processing and transforming datasets easily and efficiently.
3. Accelerate
For simplifying training across multiple GPUs, or CPUs. Easy compatibility.

#### Project
Fine-tune DialoGPT-medium on a custom dataset of prompt-completion pairs (stored in .jsonl format) to generate more aligned or domain-specific outputs.

#### JSON file
The file fine_tune_styles_dataset.jsonl contains data as shown below with 30 samples totally.
```
{"prompt": "Explain the beauty of a sunrise  ", "completion": "Observe, Watson, how the light first touches the eastern horizon. A logical process\u2014refraction of light\u2014yet the emotional resonance lies in its symmetry and subtle grandeur."}
{"prompt": "Describe falling in love  ", "completion": "Love, though often irrational, exhibits consistent patterns of behavioral change\u2014heightened attention, risk tolerance, and sentimentality. The signs are elementary, my dear Watson."}
{"prompt": "What is time?  ", "completion": "Time is the greatest variable in our deductions. It orders chaos, yet deceives. One must measure not merely the moment, but its consequence."}
{"prompt": "What does hope feel like?  ", "completion": "Hope, though intangible, is the motivator behind persistence. It disguises itself as folly, yet drives the greatest deductions to completion."}
{"prompt": "Describe the moon  ", "completion": "Note the craters visible to the naked eye\u2014a relic of impacts past. The moon tells a tale if one only chooses to observe and deduce."}
```
#### Result 
prompt = "What is darkness?\n"
Output:
What is darkness?
 darkness. The darkness is the source of all things. It binds us to it. The unseen is the deadliest. This is a mystery. An eternal. Darkness. Has no end. 7.3.4.1 ...

The answer is followed by random numbers. Sometimes followed by repeating random words. As DialogGpt is trained on Reddit data, it was also observed that there were 'memes' and certain words that would be used on social forums. 
Better models would give better results. For the limited resources available, the model performed well. 

### Chat with Groq LLM
Groq LLM with python has been used for both synchronous and asynchronous interactions with LLaMa3 models, including streaming responses and full conversation memory. 

Here model llama3-70b has been used. LLaMA (Large Language Model Meta AI) is a family of open source language models created by Meta AI. 

1. Synchronous
A blocking way of communicating with the model - the program waits until it receives a full response before doing anything else.
ex - quick scripts, simple queries, notebooks

2. Ansynchronous
A non-blocking method of communication. While waiting for the model's reply, the program can do other tasks. 
ex - web apps, scalable backends, applications requiring concurrency.

3. Streaming
The model sends back tokens as they are generated, rather than waiting to send the entire response. Live response.
ex - real-time interfaces, chatbots, terminal UIs, responsive applications.

4. Full Conversation Memory.
The application maintains a list of messages (roles: system, user, assistant) that allows the model to understand context, previous turns, tone, goals...
This is important because LLMs are stateless - they do not remember anything between calls unless we provide the previous messages.
ex - Chatbots, teaching assistants, long-form discussions, games.

### RAG with Groq, LlamaIndex, HuggingFace Embeddings
Retrieval-Augmented Generation system retrieves relevant documents and augments them into a prompt to improve LLM output. 

Retrieval-Augmented Generation (RAG) system using 
- Groq's LLaMA 3 (70B) via the Groq API — for fast, intelligent text generation.
- LlamaIndex — for loading and indexing documents, and enabling memory-aware conversations.
- HuggingFaceEmbedding — for converting text into high-dimensional vectors, enabling similarity search.
- ChatMemoryBuffer — for multi-turn dialogue with memory, giving chat interactions more human-like continuity.

#### Example Use Case
In this code, we consider a CV.pdf file as the document source.
The user can ask questions like:
```
“Is this person a good fit for a Reinforcement Learning master's thesis?”
```
```
“What interview questions would be relevant based on the CV?”
```
The system uses semantic search over the CV.pdf, pulls the most relevant information, and generates answers using the LLM.

#### Workflow
```
graph TD;
    A[User Query] --> B[Vector Search with HuggingFaceEmbedding]
    B --> C[Top-k Similar Chunks from CV.pdf]
    C --> D[LLaMA3 on Groq]
    D --> E[Answer Generated]
    E --> F[Stored in ChatMemoryBuffer]
```
Step	Description
A. User Query<br>
The user inputs a natural language question (e.g., "Is this person fit for a research role in RL?").
B. Vector Search with HuggingFaceEmbedding<br>
The question is converted into a dense vector using the HuggingFace embedding model. This allows the system to understand the meaning of the question, not just keywords.
C. Top-k Similar Chunks from CV.pdf<br>
Using the embedded query, the system searches the pre-indexed CV document for the most semantically similar sections (e.g., work experience, projects, publications).
D. LLaMA3 on Groq<br>
The retrieved chunks, along with the user query, are sent to the LLaMA 3 model running on Groq’s ultra-fast inference engine to generate a thoughtful, contextual answer.
E. Answer Generated<br>
The LLM returns an answer — informed by both the document and the user query.
F. Stored in ChatMemoryBuffer<br>
The conversation (user query + LLM answer) is stored in memory, allowing the next interaction to build upon the previous one — enabling smooth, context-aware chat.

This design lets the chatbot think over documents intelligently and hold multi-turn conversations with memory, making it ideal for tasks like document analysis, CV screening, report summarization, and more.

### Local LLM and PDF querying with Mistral 7B
Mistral 7B quantized model is used. A quantized model is recommended for local use on consumer hardware. 
Quantization reduces the precision of model weights from 16-bit or 32-bit floats (FP16/FP32) to smaller formats like 4-bit or 5-bit integers, which:
1. Reduces memory usage
2. Speeds up inference
3. Allows running on consumer GPUs or even CPUs

We run the Quantized Mistral 7B model loaded with llama-cpp-python running completely locally without needing any external APIs. This is also combined with a PDF parsing to simulate basic RAG for a single document. 

Local Running Benefits
1. No internet required. Data security. 
2. Low Latency, but depends on the system power.

PDF Querying and RAG are different:
1. RAG uses embeddings, retrieval, chunking, indexing. But PDF Querying uses the entire document directly in the prompt. 
2. RAG is scalable. But PDF Querying will hit the context/token limit quickly. 
3. RAG would have chat memory buffer while PDF querying has no memory or history tracking. 

PDF Querying is basically context stuffing / prompt injection. Not chunked, indexed, or retrieved. It is much more fragile if long documents are involved. 
