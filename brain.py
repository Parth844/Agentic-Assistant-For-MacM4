#!/usr/bin/env python3
"""
Parth AI Assistant - Ultimate Agentic Version
ReAct Architecture + Advanced Capabilities + True Autonomy
"""

import requests
import os
import sys
import subprocess
import json
import uuid
import logging
import time
import threading
import re
import tempfile
import hashlib
import ast
import warnings
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum

# Filter warnings
warnings.filterwarnings('ignore')

# Optional Dependencies with graceful degradation
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from PIL import Image
    import pytesseract
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    Image = None
    pytesseract = None

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None

# Configuration
MODEL_NAME = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MEMORY_FILE = "memory.json"
TASK_FILE = "tasks.json"
VECTOR_FILE = "vector_memory.json"
CONFIG_FILE = "config.json"
AGENT_MEMORY_FILE = "agent_memory.json"
WORKFLOW_FILE = "workflows.json"
SCREENSHOT_DIR = os.path.expanduser("~/Pictures/ParthAI")
SANDBOX_DIR = os.path.expanduser("~/.parth_sandbox")

# Ensure directories exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(SANDBOX_DIR, exist_ok=True)

# ==============================
# Logging - Less verbose
# ==============================
logging.basicConfig(
    filename="parth_ai.log",
    level=logging.WARNING,  # Only log warnings and errors
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==============================
# Startup Diagnostics
# ==============================
class Diagnostics:
    """Check what's available on this system"""
    
    def __init__(self):
        self.results = {}
        self._run_checks()
    
    def _run_checks(self):
        """Check all dependencies"""
        # Ollama
        try:
            r = requests.get("http://localhost:11434", timeout=2)
            self.results['ollama'] = r.status_code == 200
            if self.results['ollama']:
                # Check model availability
                try:
                    models = requests.get("http://localhost:11434/api/tags", timeout=2).json()
                    model_names = [m['name'] for m in models.get('models', [])]
                    self.results['models'] = model_names
                except:
                    self.results['models'] = []
        except:
            self.results['ollama'] = False
            self.results['models'] = []
        
        # Tools
        self.results['whisper'] = WHISPER_AVAILABLE
        self.results['vision'] = VISION_AVAILABLE
        self.results['audio'] = AUDIO_AVAILABLE
        self.results['numpy'] = NUMPY_AVAILABLE
        
        # macOS tools
        self.results['brightness'] = self._check_cmd("brightness")
        self.results['cliclick'] = self._check_cmd("cliclick")
        self.results['ddgr'] = self._check_cmd("ddgr")
        self.results['say'] = self._check_cmd("say")
        
        # Accessibility (for cliclick to work)
        self.results['accessibility'] = self._check_accessibility()
    
    def _check_cmd(self, cmd: str) -> bool:
        try:
            return subprocess.run(["which", cmd], capture_output=True, timeout=2).returncode == 0
        except:
            return False
    
    def _check_accessibility(self) -> bool:
        """Check if we have accessibility permissions"""
        try:
            # Try a harmless test with cliclick
            if not self.results['cliclick']:
                return False
            result = subprocess.run(["cliclick", "-V"], capture_output=True, timeout=2)
            # If it runs without accessibility warning, we have permissions
            stderr = result.stderr.decode() if result.stderr else ""
            return "Accessibility" not in stderr
        except:
            return False
    
    def print_report(self):
        """Print startup report"""
        print("\n" + "="*50)
        print("🤖 Parth AI - Startup Diagnostics")
        print("="*50)
        
        # Core
        status = "✅" if self.results['ollama'] else "❌"
        print(f"{status} Ollama Connection")
        
        if self.results['models']:
            print(f"   Available models: {', '.join(self.results['models'][:3])}")
        
        # Features
        print("\n📦 Features:")
        features = [
            ('whisper', "Voice Recognition"),
            ('vision', "Screen Analysis"),
            ('audio', "Audio Recording"),
            ('numpy', "Vector Search"),
        ]
        for key, name in features:
            status = "✅" if self.results[key] else "❌"
            print(f"  {status} {name}")
        
        # macOS Integration
        print("\n🖥️  macOS Integration:")
        mac_tools = [
            ('brightness', "Brightness Control"),
            ('cliclick', "Mouse/Keyboard Control"),
            ('accessibility', "Accessibility Permissions"),
            ('say', "Text-to-Speech"),
            ('ddgr', "Web Search"),
        ]
        for key, name in mac_tools:
            status = "✅" if self.results[key] else "❌"
            print(f"  {status} {name}")
        
        # Warnings
        if not self.results['accessibility'] and self.results['cliclick']:
            print("\n⚠️  WARNING: cliclick installed but no Accessibility permissions!")
            print("   To enable: System Preferences → Security & Privacy → Accessibility")
            print("   Add Terminal (or your Python IDE) and check the box.")
        
        if not self.results['ollama']:
            print("\n❌ ERROR: Ollama not running!")
            print("   Start with: ollama serve")
            print("   Then pull model: ollama pull llama3.1:8b")
        
        print("\n" + "="*50 + "\n")

# Run diagnostics at startup
DIAG = Diagnostics()

# ==============================
# Data Structures
# ==============================
class ActionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class Thought:
    content: str
    timestamp: float = field(default_factory=time.time)
    step_number: int = 0

@dataclass
class Action:
    tool: str
    params: Dict[str, Any]
    description: str
    status: ActionStatus = ActionStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0

@dataclass
class Observation:
    content: str
    timestamp: float = field(default_factory=time.time)
    action_ref: Optional[int] = None

# ==============================
# Configuration
# ==============================
class ConfigManager:
    def __init__(self):
        self.defaults = {
            "voice_mode": True,
            "vision_enabled": True,
            "max_iterations": 10,
            "auto_retry": True,
            "sandbox_enabled": True,
            "proactive_mode": False,
            "model_name": "llama3.1:8b",
            "context_window": 8192,
            "temperature": 0.7,
        }
        self.config = self._load()
        
    def _load(self) -> Dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                    merged = self.defaults.copy()
                    merged.update(loaded)
                    return merged
            except Exception as e:
                logging.error(f"Config load error: {e}")
        return self.defaults.copy()
    
    def save(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Config save error: {e}")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        self.config[key] = value
        self.save()

# ==============================
# LLM Client
# ==============================
class LLMClient:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model = config.get("model_name", MODEL_NAME)
        self.timeout = 60
        
        # Test connection
        self.available = self._test_connection()
    
    def _test_connection(self) -> bool:
        try:
            r = requests.get("http://localhost:11434", timeout=2)
            return r.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, system: Optional[str] = None, 
                 temperature: Optional[float] = None, json_mode: bool = False) -> str:
        if not self.available:
            return "Error: Ollama not available. Run 'ollama serve'"
        
        temp = temperature or self.config.get("temperature", 0.7)
        
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        
        for attempt in range(2):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "format": "json" if json_mode else None,
                        "options": {
                            "temperature": temp,
                            "num_ctx": self.config.get("context_window", 8192),
                            "num_predict": 1024,
                        }
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
                
            except Exception as e:
                logging.error(f"LLM error: {e}")
                if attempt == 1:
                    return f"Error: {e}"
                time.sleep(1)
        
        return "Error: Generation failed"
    
    def embed(self, text: str) -> Optional[List[float]]:
        if not self.available:
            return None
        try:
            response = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            return response.json().get("embedding")
        except Exception as e:
            logging.warning(f"Embedding failed: {e}")
            return None

# ==============================
# ReAct Agent Core
# ==============================
class ReActAgent:
    def __init__(self, llm: LLMClient, tools: Dict[str, Callable], config: ConfigManager):
        self.llm = llm
        self.tools = tools
        self.config = config
        self.max_iterations = config.get("max_iterations", 10)
        
        self.thoughts: List[Thought] = []
        self.actions: List[Action] = []
        self.observations: List[Observation] = []
        
    def solve(self, user_request: str, context: Optional[Dict] = None) -> str:
        self.thoughts = []
        self.actions = []
        self.observations = []
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            # REASON
            thought = self._reason(user_request, context)
            self.thoughts.append(thought)
            
            # Check for final answer
            if "FINAL_ANSWER:" in thought.content:
                return thought.content.split("FINAL_ANSWER:")[1].strip()
            
            # Check for clarification
            if "NEED_CLARIFICATION:" in thought.content:
                return thought.content.split("NEED_CLARIFICATION:")[1].strip()
            
            # Parse and execute action
            action = self._parse_action(thought.content)
            if not action:
                # No action found, treat as response
                return thought.content
            
            self.actions.append(action)
            
            # EXECUTE
            result = self._execute_with_retry(action)
            
            # OBSERVE
            observation = Observation(
                content=str(result)[:1000],
                action_ref=len(self.actions) - 1
            )
            self.observations.append(observation)
            action.result = result
            action.status = ActionStatus.SUCCESS if not action.error else ActionStatus.FAILED
        
        return self._summarize_results(user_request)
    
    def _reason(self, user_request: str, context: Optional[Dict]) -> Thought:
        history = self._format_history()
        tools_desc = self._format_tools()
        
        system = f"""You are Parth AI, an autonomous assistant. Solve tasks step by step.

Available Tools:
{tools_desc}

Respond in this format:
THOUGHT: your reasoning about what to do
ACTION: {{"tool": "tool_name", "params": {{"key": "value"}}, "description": "what this does"}}
OR
FINAL_ANSWER: your complete response when done
OR
NEED_CLARIFICATION: what you need to know"""

        prompt = f"""User Request: {user_request}

{history}

What should you do next? Think step by step."""
        
        response = self.llm.generate(prompt, system, temperature=0.7)
        return Thought(content=response, step_number=len(self.thoughts) + 1)
    
    def _parse_action(self, thought_content: str) -> Optional[Action]:
        if "ACTION:" not in thought_content:
            return None
        
        try:
            # Extract JSON after ACTION:
            parts = thought_content.split("ACTION:")
            if len(parts) < 2:
                return None
            
            action_text = parts[1].strip()
            
            # Handle case where there's extra text after JSON
            lines = action_text.split('\n')
            json_str = lines[0] if lines else action_text
            
            # Try to find complete JSON
            try:
                action_data = json.loads(json_str)
            except:
                # Maybe it spans multiple lines, try to find closing brace
                full_json = ""
                brace_count = 0
                for char in action_text:
                    full_json += char
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            break
                
                action_data = json.loads(full_json)
            
            return Action(
                tool=action_data.get("tool", "unknown"),
                params=action_data.get("params", {}),
                description=action_data.get("description", "Execute tool")
            )
        except Exception as e:
            logging.error(f"Failed to parse action: {e}")
            return None
    
    def _execute_with_retry(self, action: Action) -> Any:
        max_retries = 1 if self.config.get("auto_retry") else 0
        
        for attempt in range(max_retries + 1):
            try:
                action.status = ActionStatus.RUNNING
                
                if action.tool not in self.tools:
                    raise ValueError(f"Unknown tool: {action.tool}")
                
                result = self.tools[action.tool](**action.params)
                return result
                
            except Exception as e:
                action.error = str(e)
                logging.warning(f"Action failed: {e}")
                
                if attempt < max_retries:
                    action.status = ActionStatus.RETRYING
                    time.sleep(0.5)
                else:
                    action.status = ActionStatus.FAILED
                    return f"Error: {e}"
        
        return "Max retries exceeded"
    
    def _format_history(self) -> str:
        if not self.thoughts:
            return "No previous actions."
        
        lines = ["Recent actions:"]
        for i, thought in enumerate(self.thoughts[-3:]):
            lines.append(f"  Step {thought.step_number}: {thought.content[:80]}...")
        return "\n".join(lines)
    
    def _format_tools(self) -> str:
        descriptions = []
        for name in list(self.tools.keys())[:10]:  # Limit to avoid long prompts
            func = self.tools[name]
            desc = func.__doc__ or "No description"
            descriptions.append(f"- {name}: {desc.split(chr(10))[0]}")
        return "\n".join(descriptions)
    
    def _summarize_results(self, original_request: str) -> str:
        return f"Task completed after {len(self.actions)} steps. History: " + \
               ", ".join([a.tool for a in self.actions])

# ==============================
# Tool Implementations
# ==============================
class ToolRegistry:
    def __init__(self, agent: 'ParthAgent'):
        self.agent = agent
        self.diag = DIAG
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Callable]:
        tools = {}
        
        # Only register if dependencies available
        tools['read_file'] = self.read_file
        tools['write_file'] = self.write_file
        tools['list_files'] = self.list_files
        tools['analyze_code'] = self.analyze_code
        
        if self.diag.results['brightness']:
            tools['set_brightness'] = self.set_brightness
        
        tools['set_volume'] = self.set_volume
        tools['take_screenshot'] = self.take_screenshot
        tools['lock_screen'] = self.lock_screen
        tools['open_application'] = self.open_application
        tools['get_system_info'] = self.get_system_info
        
        if self.diag.results['vision']:
            tools['analyze_screen'] = self.analyze_screen
            if self.diag.results['cliclick'] and self.diag.results['accessibility']:
                tools['click_on_text'] = self.click_on_text
        
        if self.diag.results['ddgr']:
            tools['web_search'] = self.web_search
        
        tools['remember_fact'] = self.remember_fact
        tools['recall_facts'] = self.recall_facts
        tools['add_task'] = self.add_task
        tools['list_tasks'] = self.list_tasks
        tools['ask_user'] = self.ask_user
        tools['notify'] = self.notify
        
        return tools
    
    def read_file(self, path: str, offset: int = 0, limit: int = 100) -> str:
        """Read file content"""
        try:
            if not os.path.exists(path):
                # Try relative to sandbox
                path = os.path.join(SANDBOX_DIR, path)
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                selected = lines[offset:offset+limit]
                content = ''.join(selected)
                
                self.agent.active_document = {
                    'path': path,
                    'content': ''.join(lines),
                    'total_lines': len(lines)
                }
                
                header = f"File: {path} (lines {offset+1}-{min(offset+limit, len(lines))} of {len(lines)})\n"
                return header + content
        except Exception as e:
            return f"Error reading {path}: {e}"
    
    def write_file(self, path: str, content: str, append: bool = False) -> str:
        """Write or append to file"""
        try:
            # Write to sandbox for safety
            if not path.startswith('/') and not path.startswith('~'):
                path = os.path.join(SANDBOX_DIR, path)
            
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {path} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing {path}: {e}"
    
    def list_files(self, directory: str = ".", pattern: str = "*") -> str:
        """List files in directory"""
        try:
            import glob
            if directory == ".":
                directory = SANDBOX_DIR
            
            files = glob.glob(os.path.join(directory, pattern))
            files = [f for f in files if os.path.isfile(f)]
            
            if not files:
                return f"No files found in {directory} matching '{pattern}'"
            
            result = [f"Found {len(files)} files:"]
            for f in files[:20]:  # Limit output
                size = os.path.getsize(f)
                result.append(f"  {f} ({size} bytes)")
            
            if len(files) > 20:
                result.append(f"  ... and {len(files) - 20} more")
            
            return "\n".join(result)
        except Exception as e:
            return f"Error: {e}"
    
    def analyze_code(self, file_path: str) -> str:
        """Analyze code file for issues"""
        content = self.read_file(file_path)
        if content.startswith("Error"):
            return content
        
        code = '\n'.join(content.split('\n')[1:])  # Remove header
        
        # Simple analysis without heavy LLM call
        issues = []
        
        # Check for common issues
        if 'except:' in code:
            issues.append("Line with bare 'except:' - should catch specific exceptions")
        if 'print(' in code and 'logging' not in code:
            issues.append("Uses print() instead of logging")
        
        # Try to parse
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        if issues:
            return f"Code Analysis for {file_path}:\n" + "\n".join(f"- {i}" for i in issues)
        return f"Code in {file_path} looks good! No obvious issues found."
    
    def set_brightness(self, level: float) -> str:
        """Set screen brightness 0.0-1.0"""
        if not self.diag.results['brightness']:
            return "Brightness control not available. Install: brew install brightness"
        try:
            subprocess.run(["brightness", str(level)], check=True, capture_output=True)
            return f"Brightness set to {level:.0%}"
        except Exception as e:
            return f"Failed: {e}"
    
    def set_volume(self, level: int) -> str:
        """Set system volume 0-100"""
        try:
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"], 
                          check=True, capture_output=True)
            return f"Volume set to {level}%"
        except Exception as e:
            return f"Error: {e}"
    
    def take_screenshot(self, region: Optional[str] = None) -> str:
        """Capture screenshot"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"ss_{timestamp}.png")
            
            cmd = ["screencapture", "-x"]
            if region:
                cmd.extend(["-R", region])
            cmd.append(path)
            
            subprocess.run(cmd, check=True, capture_output=True)
            return f"Screenshot saved: {path}"
        except Exception as e:
            return f"Error: {e}"
    
    def lock_screen(self) -> str:
        """Lock the screen"""
        try:
            subprocess.run(["pmset", "displaysleepnow"], check=True, capture_output=True)
            return "Screen locked"
        except Exception as e:
            return f"Error: {e}"
    
    def open_application(self, app_name: str) -> str:
        """Open macOS application"""
        try:
            subprocess.run(["open", "-a", app_name], check=True, capture_output=True)
            return f"Opened {app_name}"
        except Exception as e:
            return f"Error: {e}"
    
    def get_system_info(self, info_type: str = "battery") -> str:
        """Get system information"""
        try:
            if info_type == "battery":
                result = subprocess.run(["pmset", "-g", "batt"], 
                                    capture_output=True, text=True, check=True)
                return result.stdout.strip()
            elif info_type == "cpu":
                result = subprocess.run(["top", "-l", "1", "-n", "0"], 
                                    capture_output=True, text=True, check=True)
                return "\n".join(result.stdout.split('\n')[:5])
            return "Unknown info type. Use: battery, cpu"
        except Exception as e:
            return f"Error: {e}"
    
    def analyze_screen(self, question: Optional[str] = None) -> str:
        """Analyze screen with OCR"""
        if not self.diag.results['vision']:
            return "Vision not available. Install: brew install tesseract && pip install pytesseract pillow"
        
        try:
            # Take screenshot
            path = self.take_screenshot().replace("Screenshot saved: ", "")
            if not os.path.exists(path):
                return "Failed to capture screen"
            
            # OCR
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
            
            # Analyze
            prompt = f"""Screen shows this text:
{text[:1500]}

Question: {question or "What do you see and what should the user do?"}

Provide brief analysis."""
            
            return self.agent.llm.generate(prompt)
        except Exception as e:
            return f"Vision error: {e}"
    
    def click_on_text(self, text: str) -> str:
        """Click on text on screen"""
        if not self.diag.results['vision']:
            return "Vision not available"
        if not self.diag.results['cliclick']:
            return "cliclick not installed. Run: brew install cliclick"
        if not self.diag.results['accessibility']:
            return "No Accessibility permissions! Go to System Preferences → Security & Privacy → Accessibility → Add Terminal"
        
        try:
            # Screenshot and find text
            path = self.take_screenshot().replace("Screenshot saved: ", "")
            img = Image.open(path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            for i, word in enumerate(data['text']):
                if text.lower() in word.lower():
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    
                    subprocess.run(["cliclick", f"c:{x},{y}"], check=True, capture_output=True)
                    return f"Clicked '{text}' at ({x}, {y})"
            
            return f"Text '{text}' not found on screen"
        except Exception as e:
            return f"Click failed: {e}"
    
    def web_search(self, query: str, num_results: int = 3) -> str:
        """Search web using ddgr"""
        if not self.diag.results['ddgr']:
            return "Web search not available. Install: brew install ddgr"
        
        try:
            result = subprocess.run(
                ["ddgr", "--json", "-n", str(num_results), query],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                summaries = []
                for item in data[:num_results]:
                    title = item.get('title', 'No title')
                    abstract = item.get('abstract', 'No description')[:150]
                    summaries.append(f"• {title}: {abstract}...")
                return "\n".join(summaries)
            else:
                return f"Search failed: {result.stderr[:200]}"
        except Exception as e:
            return f"Search error: {e}"
    
    def remember_fact(self, category: str, fact: str) -> str:
        """Store fact in memory"""
        self.agent.long_term_memory.learn_fact(category, fact)
        return f"✓ Remembered: [{category}] {fact}"
    
    def recall_facts(self, category: Optional[str] = None) -> str:
        """Recall facts from memory"""
        memory = self.agent.long_term_memory
        
        if category:
            facts = memory.facts.get(category, [])
            if facts:
                return f"Facts about {category}:\n" + "\n".join(f"- {f['fact']}" for f in facts[-5:])
            return f"No facts about {category}"
        
        return memory.get_context()
    
    def add_task(self, text: str, priority: str = "normal") -> str:
        """Add a task"""
        return self.agent.task_engine.add(text, priority)
    
    def list_tasks(self) -> str:
        """List all tasks"""
        return self.agent.task_engine.list()
    
    def ask_user(self, question: str) -> str:
        """Ask user for input"""
        return f"ASK_USER:{question}"
    
    def notify(self, message: str, title: str = "Parth AI") -> str:
        """Send macOS notification"""
        try:
            subprocess.run([
                "osascript", "-e",
                f'display notification "{message}" with title "{title}"'
            ], check=True, capture_output=True)
            return "Notification sent"
        except Exception as e:
            return f"Error: {e}"

# ==============================
# Long-term Memory
# ==============================
class LongTermMemory:
    def __init__(self):
        self.file_path = AGENT_MEMORY_FILE
        self.facts: Dict[str, List[Dict]] = {}
        self.preferences: Dict[str, Any] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.facts = data.get('facts', {})
                    self.preferences = data.get('preferences', {})
            except Exception as e:
                logging.error(f"Memory load error: {e}")
    
    def save(self):
        try:
            with open(self.file_path, 'w') as f:
                json.dump({
                    'facts': self.facts,
                    'preferences': self.preferences,
                    'updated': time.time()
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Memory save error: {e}")
    
    def learn_fact(self, category: str, fact: str):
        if category not in self.facts:
            self.facts[category] = []
        
        # Avoid duplicates
        existing = [f['fact'] for f in self.facts[category]]
        if fact not in existing:
            self.facts[category].append({
                'fact': fact,
                'learned_at': time.time()
            })
            self.save()
    
    def set_preference(self, key: str, value: Any):
        self.preferences[key] = {
            'value': value,
            'set_at': time.time()
        }
        self.save()
    
    def get_context(self) -> str:
        parts = []
        for key, data in self.preferences.items():
            parts.append(f"Pref: {key}={data['value']}")
        
        for cat, facts in self.facts.items():
            recent = [f['fact'] for f in facts[-2:]]
            if recent:
                parts.append(f"{cat}: {', '.join(recent)}")
        
        return "\n".join(parts) if parts else "No prior context."

# ==============================
# Task Engine
# ==============================
class TaskEngine:
    def __init__(self):
        self.tasks: List[Dict] = []
        self.file_path = TASK_FILE
        self._load()
    
    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.tasks = json.load(f)
            except:
                self.tasks = []
    
    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.tasks, f, indent=2)
    
    def add(self, text: str, priority: str = "normal") -> str:
        task = {
            'id': str(uuid.uuid4())[:8],
            'text': text,
            'priority': priority,
            'done': False,
            'created': time.time()
        }
        self.tasks.append(task)
        self.save()
        return f"Added task: {text} ({priority})"
    
    def list(self) -> str:
        active = [t for t in self.tasks if not t.get('done')]
        if not active:
            return "No active tasks."
        
        lines = [f"Active tasks ({len(active)}):"]
        for t in sorted(active, key=lambda x: x['created'], reverse=True)[:10]:
            emoji = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(t['priority'], "⚪")
            lines.append(f"{emoji} {t['text']}")
        return "\n".join(lines)

# ==============================
# Voice & Audio
# ==============================
class VoiceManager:
    def __init__(self, config: ConfigManager, diag: Diagnostics):
        self.config = config
        self.diag = diag
        self.voice_mode = config.get("voice_mode", True) and diag.results['say']
    
    def speak(self, text: str):
        if not self.voice_mode or not text:
            return
        try:
            # Truncate long text
            text = text[:300] + "..." if len(text) > 300 else text
            subprocess.run(["say", text], check=False, capture_output=True)
        except:
            pass

class AudioProcessor:
    def __init__(self, diag: Diagnostics):
        self.model = None
        self.available = diag.results['whisper'] and diag.results['audio']
    
    def load_model(self):
        if not self.available or self.model:
            return self.model
        
        try:
            self.model = whisper.load_model("base")
        except Exception as e:
            logging.error(f"Whisper load failed: {e}")
            self.available = False
        
        return self.model
    
    def record(self, duration: int = 5) -> Optional[Any]:
        if not self.available:
            return None
        
        try:
            recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            return recording.flatten()
        except Exception as e:
            logging.error(f"Recording failed: {e}")
            return None
    
    def transcribe(self, audio: Any) -> Optional[str]:
        model = self.load_model()
        if not model or audio is None:
            return None
        
        try:
            result = model.transcribe(audio, fp16=False, language="en")
            return result.get("text", "").strip()
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return None

# ==============================
# Main Parth Agent
# ==============================
class ParthAgent:
    def __init__(self):
        print("Initializing Parth AI...")
        
        self.config = ConfigManager()
        self.diag = DIAG
        self.llm = LLMClient(self.config)
        self.long_term_memory = LongTermMemory()
        self.task_engine = TaskEngine()
        self.voice = VoiceManager(self.config, self.diag)
        self.audio = AudioProcessor(self.diag)
        
        # Tool registry
        self.tool_registry = ToolRegistry(self)
        self.tools = self.tool_registry.tools
        
        # ReAct agent
        self.react = ReActAgent(self.llm, self.tools, self.config)
        
        # State
        self.active_document: Optional[Dict] = None
        
        print(f"✓ Loaded {len(self.tools)} tools")
        print(f"✓ Memory: {sum(len(f) for f in self.long_term_memory.facts.values())} facts")
        print("Ready!\n")
        
    def process(self, user_input: str) -> str:
        logging.info(f"Processing: {user_input}")
        
        # Build context
        context = {
            'facts': self.long_term_memory.get_context(),
            'active_doc': self.active_document['path'] if self.active_document else None
        }
        
        # Use ReAct
        response = self.react.solve(user_input, context)
        
        # Extract learnings
        self._extract_learnings(user_input, response)
        
        return response
    
    def _extract_learnings(self, user_input: str, response: str):
        """Extract facts from conversation"""
        patterns = [
            (r'my name is (\w+)', 'name'),
            (r'i (?:work|am) as (?:a|an)? (.+)', 'profession'),
            (r'i (?:like|love|prefer) (.+)', 'preferences'),
        ]
        
        for pattern, category in patterns:
            match = re.search(pattern, user_input, re.I)
            if match:
                fact = match.group(1).strip().rstrip('.')
                if len(fact) > 2:
                    self.long_term_memory.learn_fact(category, fact)

# ==============================
# Modern GUI
# ==============================
import tkinter as tk
from tkinter import scrolledtext
import queue

class ParthGUI:
    def __init__(self):
        print("Starting GUI...")
        
        self.agent = ParthAgent()
        self.root = tk.Tk()
        self.root.title("Parth AI - Agentic Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0a0a0a")
        
        self.colors = {
            'bg': '#0a0a0a', 'sidebar': '#111111', 'surface': '#1a1a1a',
            'border': '#333333', 'text': '#ffffff', 'secondary': '#888888',
            'accent': '#10a37f', 'accent_hover': '#0d8c6d',
            'user': '#1e3a2f', 'ai': '#262626', 'error': '#ff4444'
        }
        
        self.queue = queue.Queue()
        self.is_processing = False
        
        self._setup_ui()
        self._show_welcome()
        self._process_queue()
        
    def _setup_ui(self):
        # Main container
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        sidebar = tk.Frame(main, bg=self.colors['sidebar'], width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Logo
        tk.Label(sidebar, text="◉ Parth AI", bg=self.colors['sidebar'],
                fg=self.colors['accent'], font=('SF Pro', 18, 'bold')).pack(pady=20)
        
        # Status
        self.status_label = tk.Label(sidebar, text="● Ready", 
                                    bg=self.colors['sidebar'],
                                    fg=self.colors['accent'],
                                    font=('SF Pro', 11))
        self.status_label.pack()
        
        # Tool count
        tool_count = len(self.agent.tools)
        tk.Label(sidebar, text=f"{tool_count} tools available",
                bg=self.colors['sidebar'],
                fg=self.colors['secondary'],
                font=('SF Pro', 10)).pack(pady=5)
        
        # Quick Actions
        tk.Label(sidebar, text="Quick Actions", bg=self.colors['sidebar'],
                fg=self.colors['secondary'], font=('SF Pro', 12)).pack(pady=(30,10))
        
        actions = [
            ("👁️  Analyze Screen", lambda: self._quick("what's on my screen")),
            ("📸 Screenshot", lambda: self._quick("take a screenshot")),
            ("🔒 Lock Mac", lambda: self._quick("lock screen")),
            ("📋 List Tasks", lambda: self._quick("list my tasks")),
        ]
        
        for text, cmd in actions:
            btn = tk.Button(sidebar, text=text, bg=self.colors['surface'],
                          fg=self.colors['text'], relief=tk.FLAT,
                          font=('SF Pro', 11), cursor='hand2',
                          command=cmd)
            btn.pack(fill=tk.X, padx=15, pady=3)
        
        # Main content
        content = tk.Frame(main, bg=self.colors['bg'])
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Header
        header = tk.Frame(content, bg=self.colors['surface'], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="Agent Conversation", bg=header.cget('bg'),
                fg=self.colors['text'], font=('SF Pro', 14, 'bold')).pack(side=tk.LEFT, padx=20, pady=12)
        
        # Chat area
        self.chat_text = scrolledtext.ScrolledText(
            content,
            wrap=tk.WORD,
            bg=self.colors['bg'],
            fg=self.colors['text'],
            font=('SF Mono', 12),
            padx=20, pady=20,
            relief=tk.FLAT,
            highlightthickness=0
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chat_text.config(state=tk.DISABLED)
        
        # Configure tags
        self.chat_text.tag_config("user", foreground="#10a37f", font=('SF Mono', 12, 'bold'))
        self.chat_text.tag_config("ai", foreground="#ffffff", font=('SF Mono', 12))
        self.chat_text.tag_config("system", foreground="#888888", font=('SF Mono', 11, 'italic'))
        self.chat_text.tag_config("error", foreground="#ff4444", font=('SF Mono', 11))
        
        # Input area
        input_frame = tk.Frame(content, bg=self.colors['surface'], height=70)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)
        input_frame.pack_propagate(False)
        
        input_container = tk.Frame(input_frame, bg=input_frame.cget('bg'))
        input_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        self.input_field = tk.Entry(
            input_container,
            bg=self.colors['bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief=tk.FLAT,
            font=('SF Pro', 13),
            highlightthickness=1,
            highlightcolor=self.colors['accent']
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_field.bind("<Return>", self._send)
        self.input_field.focus()
        
        # Voice button (only if available)
        if self.agent.diag.results['whisper'] and self.agent.diag.results['audio']:
            self.voice_btn = tk.Button(
                input_container,
                text="🎤",
                bg=self.colors['bg'],
                fg=self.colors['text'],
                relief=tk.FLAT,
                font=('SF Pro', 14),
                command=self._voice_input
            )
            self.voice_btn.pack(side=tk.LEFT, padx=(10, 5))
        else:
            self.voice_btn = None
        
        # Send button
        self.send_btn = tk.Button(
            input_container,
            text="Send →",
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            font=('SF Pro', 11, 'bold'),
            cursor='hand2',
            command=self._send
        )
        self.send_btn.pack(side=tk.RIGHT)
        
        # Hover effect
        self.send_btn.bind("<Enter>", lambda e: self.send_btn.config(bg=self.colors['accent_hover']))
        self.send_btn.bind("<Leave>", lambda e: self.send_btn.config(bg=self.colors['accent']))
    
    def _show_welcome(self):
        welcome = """👋 Welcome to Parth AI!

I'm an autonomous assistant that can think, plan, and execute tasks.

Try asking me:
• "Analyze what's on my screen"
• "List files and analyze the code"
• "Set brightness to 50% and take a screenshot"
• "Remember that I prefer dark mode"

I'll break down complex tasks into steps and execute them."""
        
        self._add_message("system", welcome)
        
        if self.agent.diag.results['say']:
            self.agent.voice.speak("Hello! I'm Parth. How can I help you?")
    
    def _quick(self, command: str):
        self.input_field.delete(0, tk.END)
        self.input_field.insert(0, command)
        self._send()
    
    def _add_message(self, role: str, content: str):
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, "\n")
        
        if role == "user":
            self.chat_text.insert(tk.END, f"You: {content}\n", "user")
        elif role == "ai":
            self.chat_text.insert(tk.END, f"Parth: {content}\n", "ai")
        elif role == "error":
            self.chat_text.insert(tk.END, f"Error: {content}\n", "error")
        else:
            self.chat_text.insert(tk.END, f"{content}\n", "system")
        
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)
    
    def _send(self, event=None):
        text = self.input_field.get().strip()
        if not text or self.is_processing:
            return
        
        self.input_field.delete(0, tk.END)
        self._add_message("user", text)
        
        self.is_processing = True
        self.status_label.config(text="● Thinking...", fg="#ffa500")
        self.send_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self._process, args=(text,), daemon=True).start()
    
    def _process(self, text: str):
        try:
            response = self.agent.process(text)
            self.queue.put(("response", response))
        except Exception as e:
            logging.error(f"Processing error: {e}")
            self.queue.put(("error", str(e)))
        finally:
            self.queue.put(("done", None))
    
    def _voice_input(self):
        if not self.voice_btn:
            return
        
        self.voice_btn.config(bg=self.colors['accent'])
        self._add_message("system", "🎤 Listening...")
        self.status_label.config(text="● Listening...", fg="#ffa500")
        
        def process():
            try:
                audio = self.agent.audio.record(duration=5)
                if audio is not None:
                    text = self.agent.audio.transcribe(audio)
                    if text:
                        self.queue.put(("voice_input", text))
                    else:
                        self.queue.put(("error", "Could not understand audio"))
                else:
                    self.queue.put(("error", "Recording failed"))
            except Exception as e:
                self.queue.put(("error", f"Voice error: {e}"))
            finally:
                self.queue.put(("voice_done", None))
        
        threading.Thread(target=process, daemon=True).start()
    
    def _process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == "response":
                    self._add_message("ai", data)
                    self.agent.voice.speak(data)
                elif msg_type == "error":
                    self._add_message("error", data)
                elif msg_type == "voice_input":
                    self.input_field.delete(0, tk.END)
                    self.input_field.insert(0, data)
                    self._send()
                elif msg_type == "voice_done":
                    if self.voice_btn:
                        self.voice_btn.config(bg=self.colors['bg'])
                elif msg_type == "done":
                    self.is_processing = False
                    self.status_label.config(text="● Ready", fg=self.colors['accent'])
                    self.send_btn.config(state=tk.NORMAL)
                
        except queue.Empty:
            pass
        
        self.root.after(100, self._process_queue)
    
    def run(self):
        self.root.mainloop()

# ==============================
# CLI Mode
# ==============================
def run_cli():
    agent = ParthAgent()
    print("\nParth AI - Agentic Mode")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input:
                if agent.diag.results['whisper'] and agent.diag.results['audio']:
                    print("🎤 Listening...")
                    audio = agent.audio.record(duration=5)
                    if audio:
                        user_input = agent.audio.transcribe(audio) or ""
                        print(f"You (voice): {user_input}")
            
            if user_input:
                print("🤔 Thinking...")
                response = agent.process(user_input)
                print(f"\nParth: {response}")
                agent.voice.speak(response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

# ==============================
# Main Entry
# ==============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--skip-diag", action="store_true", help="Skip diagnostics")
    args = parser.parse_args()
    
    # Run diagnostics unless skipped
    if not args.skip_diag:
        DIAG.print_report()
        
        if not DIAG.results['ollama']:
            print("❌ Cannot start: Ollama not available")
            print("Start Ollama with: ollama serve")
            sys.exit(1)
    
    if args.cli:
        run_cli()
    else:
        try:
            ParthGUI().run()
        except Exception as e:
            print(f"\n❌ GUI Error: {e}")
            print("Try CLI mode: python brain.py --cli")