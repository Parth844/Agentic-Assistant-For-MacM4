# 🤖 Parth AI – Autonomous Agentic Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-Ollama-green)
![Architecture](https://img.shields.io/badge/Architecture-ReAct-orange)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🧠 Overview

**Parth AI** is a fully autonomous agentic assistant built using the **ReAct (Reason + Act)** architecture and powered by local LLMs via **Ollama**.

Unlike traditional chatbots, this system:

- 🧠 Thinks step-by-step  
- 🛠 Selects tools dynamically  
- ⚡ Executes real system actions  
- 🗃 Maintains long-term memory  
- 🎤 Supports voice interaction  
- 👁 Supports screen analysis (OCR)  
- 💻 Runs fully locally  

This is a true AI agent — not just a text generator.

---

## 🏗 Architecture
  User Input
  ↓
  ReAct Reasoning (THOUGHT)
  ↓
  Tool Selection (ACTION)
  ↓
  Execution
  ↓
  Observation
  ↓
  FINAL_ANSWER### Core Modules

- LLM Client (Ollama – LLaMA 3.1)
- ReAct Agent Loop
- Tool Registry
- Long-Term Memory Engine
- Task Manager
- Voice (Whisper + TTS)
- Vision (OCR + Screenshot)
- macOS System Control
- GUI + CLI Modes

---

## ✨ Features

### 🤖 Agentic Reasoning
- Multi-step planning  
- Automatic tool execution  
- Retry handling  
- Structured thought/action loop  

### 🧠 Persistent Memory
- Learns user preferences  
- Stores facts  
- Saves across sessions  

### 📋 Task Engine
- Add tasks  
- List tasks  
- Priority system  

### 👁 Vision
- Screenshot capture  
- OCR-based screen analysis  
- Click automation (macOS)  

### 🎤 Voice
- Audio recording  
- Whisper transcription  
- Text-to-speech responses  

### 🖥 macOS Automation
- Set brightness  
- Adjust volume  
- Lock screen  
- Open applications  
- Send notifications  
- Get battery / CPU info  

---

## 📂 Project Structure
brain.py
config.json
memory.json
tasks.json
agent_memory.json
parth_ai.log

---

## ⚙️ Installation

### 1️⃣ Install Ollama

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b

### 2️⃣ Create Virtual Environment
  python -m venv venv
source venv/bin/activate

3️⃣ Install Python Dependencies
pip install requests numpy pillow pytesseract sounddevice openai-whisper
Optional macOS tools
brew install tesseract
brew install cliclick
brew install brightness
brew install ddgr

Running Assistant:
gui:
python brain.py

cli:
python brain.py --cli


example commands:-
Analyze what's on my screen
Take a screenshot
Set brightness to 50%
Open Safari
Remember that I prefer dark mode
Add task: Finish ML project (high)
🔐 Security
	•	Local LLM execution
	•	File writes restricted to sandbox directory
	•	Accessibility permissions required for automation
	•	No cloud API dependency

⸻

🚀 Why This Project Is Advanced

Most AI assistants:
	•	Only generate text.

Parth AI:
	•	Thinks
	•	Plans
	•	Executes tools
	•	Controls system
	•	Maintains persistent memory

This is a real Agentic AI System.

⸻

🔮 Future Improvements
	•	Vector memory search
	•	Multi-agent collaboration
	•	Workflow automation builder
	•	Docker deployment
	•	Web dashboard
	•	Cross-platform support

⸻

👨‍💻 Author

Parth Tyagi
AI Systems Builder
Focused on Machine Learning, Agent Architectures, and Automation

⸻

📜 License

MIT License


---

If you want, next I can:

- Make it recruiter-optimized  
- Add a demo GIF section  
- Add architecture diagram  
- Make it look like top GitHub AI repos  

Just tell me 🔥
