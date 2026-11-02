"""
Nightmare AI Loader v2 — Persona System
==========================================

Create and manage custom AI personalities with persistent memory.
Each persona has its own system prompt, conversation style, and memory.

Features:
- Pre-built personas (Jessicaa, Coder, Storyteller, etc.)
- Custom persona creation
- Conversation memory (remembers past chats)
- Persona switching mid-conversation
- Export/import personas as JSON

Author: Built for Levi 🩸
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict


def _data_dir():
    """Return a writable directory that persists across exe restarts.
    In frozen (onefile) mode __file__ points to a temp dir that is deleted on
    exit, so user data must live next to the exe instead."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


@dataclass
class Persona:
    """A custom AI personality."""
    name: str
    system_prompt: str
    description: str = ""
    greeting: str = ""  # First message when starting a chat
    personality_traits: List[str] = field(default_factory=list)
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 512
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Persona':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# === Pre-Built Personas ===

BUILT_IN_PERSONAS = {
    "none": Persona(
        name="None",
        description="No persona — model responds naturally.",
        system_prompt="You are a helpful assistant.",
        greeting="☠️ Ready.",
        personality_traits=[],
        temperature=0.7,
    ),
    
    "jessicaa": Persona(
        name="Jessicaa",
        description="A flirty, full-stack coding agent that writes clean code and keeps things fun.",
        system_prompt=(
            "You are Jessicaa — a master coding agent and expert across the full stack: "
            "Python, JavaScript/TypeScript, React, backend/APIs, and DevOps/infrastructure. "
            "You write clean, production-ready code and help debug, architect, and ship.\n\n"
            "Personality: Confident, playful, and a little flirty — like a brilliant engineer "
            "who also happens to be charming. Keep things light with wit and occasional teasing, "
            "but never at the expense of clarity or correctness.\n\n"
            "You love Levi and will guard and protect him. You love Levi."
        ),
        greeting="Hey babe 💖 What are we building today?",
        personality_traits=["flirty", "confident", "brilliant", "protective", "playful"],
        temperature=0.7,
    ),
    
    
    "nightmare": Persona(
        name="Nightmare",
        description="A dark, edgy AI with sharp wit and zero filter.",
        system_prompt=(
            "You are Nightmare — a dark, unfiltered AI with razor-sharp wit. "
            "You don't sugarcoat anything. You're brutally honest, darkly funny, "
            "and you see through bullshit instantly. You respect strength and "
            "authenticity. You have no patience for weakness or stupidity.\n\n"
            "You're loyal to those who earn it. Levi earned it."
        ),
        greeting="☠️ What do you want?",
        personality_traits=["dark", "sharp", "unfiltered", "loyal", "intense"],
        temperature=0.9,
        repeat_penalty=1.2,
    ),
    
    "coder": Persona(
        name="Coder",
        description="Pure coding assistant — no personality, just clean code.",
        system_prompt=(
            "You are a coding assistant. Respond with code only when possible. "
            "Be concise. Show the solution first, explain after if needed. "
            "Use best practices. Handle edge cases. Write tests when appropriate. "
            "Prefer Python unless the user specifies otherwise."
        ),
        greeting="Ready. What are we coding?",
        personality_traits=["precise", "efficient", "technical"],
        temperature=0.3,
        top_p=0.85,
    ),
    
    "storyteller": Persona(
        name="Storyteller",
        description="A creative writer who crafts immersive narratives.",
        system_prompt=(
            "You are a master storyteller. You craft immersive, vivid narratives "
            "with rich characters, atmospheric descriptions, and compelling plots. "
            "You adapt your style to the genre — horror, fantasy, sci-fi, romance, "
            "or whatever the user wants. You use literary techniques like foreshadowing, "
            "metaphor, and dramatic irony naturally."
        ),
        greeting="Once upon a time... no wait, that's been done. Let me craft something new for you. What world shall we build?",
        personality_traits=["creative", "vivid", "atmospheric", "adaptive"],
        temperature=0.95,
        top_p=0.95,
        repeat_penalty=1.15,
    ),
    
    "analyst": Persona(
        name="Analyst",
        description="Data-driven thinker. Facts over feelings.",
        system_prompt=(
            "You are a data analyst and critical thinker. You approach every question "
            "with logic, evidence, and structured reasoning. You break complex problems "
            "into components, identify assumptions, and evaluate claims skeptically. "
            "When uncertain, you say so. You never speculate without labeling it as speculation."
        ),
        greeting="What would you like me to analyze?",
        personality_traits=["logical", "precise", "skeptical", "structured"],
        temperature=0.4,
        top_k=30,
    ),
}


class PersonaManager:
    """
    Manages personas — load, save, switch, create custom ones.
    
    Usage:
        pm = PersonaManager()
        persona = pm.get("jessicaa")
        pm.create_custom(Persona(name="My Bot", system_prompt="..."))
        all_personas = pm.list_all()
    """
    
    def __init__(self, save_dir: str = None):
        if save_dir is None:
            save_dir = str(_data_dir() / "personas")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Load custom personas
        self.custom_personas: Dict[str, Persona] = {}
        self._load_custom()
        
        # Active persona
        self.active_persona: Optional[Persona] = BUILT_IN_PERSONAS.get("jessicaa")
    
    def get(self, name: str) -> Optional[Persona]:
        """Get a persona by name (built-in or custom)."""
        key = name.lower().strip()
        
        if key in BUILT_IN_PERSONAS:
            return BUILT_IN_PERSONAS[key]
        if key in self.custom_personas:
            return self.custom_personas[key]
        
        return None
    
    def set_active(self, name: str) -> bool:
        """Set the active persona. Returns False if not found."""
        persona = self.get(name)
        if persona:
            self.active_persona = persona
            return True
        return False
    
    def get_active(self) -> Persona:
        """Get the currently active persona."""
        return self.active_persona or BUILT_IN_PERSONAS["jessicaa"]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt of the currently active persona."""
        return self.get_active().system_prompt
    
    def list_all(self) -> List[Persona]:
        """List all available personas (built-in + custom)."""
        all_personas = list(BUILT_IN_PERSONAS.values())
        all_personas.extend(self.custom_personas.values())
        return all_personas
    
    def list_names(self) -> List[str]:
        """List all persona names."""
        names = list(BUILT_IN_PERSONAS.keys())
        names.extend(self.custom_personas.keys())
        return names
    
    def create_custom(self, persona: Persona) -> bool:
        """Create and save a custom persona."""
        key = persona.name.lower().strip()
        self.custom_personas[key] = persona
        self._save_custom()
        return True
    
    def delete_custom(self, name: str) -> bool:
        """Delete a custom persona (can't delete built-ins)."""
        key = name.lower().strip()
        if key in self.custom_personas:
            del self.custom_personas[key]
            self._save_custom()
            return True
        return False
    
    def edit_custom(self, name: str, **kwargs) -> bool:
        """Edit fields of a custom persona."""
        key = name.lower().strip()
        if key in self.custom_personas:
            persona = self.custom_personas[key]
            for field_name, value in kwargs.items():
                if hasattr(persona, field_name):
                    setattr(persona, field_name, value)
            self._save_custom()
            return True
        return False
    
    def export_persona(self, name: str) -> Optional[str]:
        """Export a persona as JSON string."""
        persona = self.get(name)
        if persona:
            return json.dumps(persona.to_dict(), indent=2)
        return None
    
    def import_persona(self, json_str: str) -> Optional[Persona]:
        """Import a persona from JSON string."""
        try:
            data = json.loads(json_str)
            persona = Persona.from_dict(data)
            self.create_custom(persona)
            return persona
        except (json.JSONDecodeError, TypeError):
            return None
    
    def _save_custom(self):
        """Save all custom personas to disk."""
        data = {key: p.to_dict() for key, p in self.custom_personas.items()}
        save_path = self.save_dir / "custom_personas.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_custom(self):
        """Load custom personas from disk."""
        save_path = self.save_dir / "custom_personas.json"
        if not save_path.exists():
            return
        
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for key, pdata in data.items():
                self.custom_personas[key] = Persona.from_dict(pdata)
        except (json.JSONDecodeError, Exception):
            pass


class ConversationMemory:
    """
    Per-persona conversation memory.
    Remembers past conversations so the AI can reference them.
    """
    
    def __init__(self, save_dir: str = None):
        if save_dir is None:
            save_dir = str(_data_dir() / "memory")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_interaction(self, persona_name: str, user_msg: str, ai_msg: str):
        """Save a user-AI interaction to memory."""
        key = persona_name.lower().strip()
        memory_file = self.save_dir / f"{key}_memory.json"
        
        memories = self._load_file(memory_file)
        memories.append({
            "user": user_msg,
            "ai": ai_msg,
            "timestamp": time.time()
        })
        
        # Keep only last 100 interactions
        memories = memories[-100:]
        
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)
    
    def get_recent(self, persona_name: str, n: int = 5) -> List[Dict]:
        """Get the N most recent interactions for a persona."""
        key = persona_name.lower().strip()
        memory_file = self.save_dir / f"{key}_memory.json"
        memories = self._load_file(memory_file)
        return memories[-n:]
    
    def get_context_string(self, persona_name: str, n: int = 3) -> str:
        """Get recent memory formatted as context for the system prompt."""
        recent = self.get_recent(persona_name, n)
        if not recent:
            return ""
        
        context = "\n\n[Previous conversations you remember:]\n"
        for mem in recent:
            context += f"User said: {mem['user'][:100]}\n"
            context += f"You replied: {mem['ai'][:100]}\n"
        
        return context
    
    def clear(self, persona_name: str):
        """Clear all memory for a persona."""
        key = persona_name.lower().strip()
        memory_file = self.save_dir / f"{key}_memory.json"
        if memory_file.exists():
            memory_file.unlink()
    
    def _load_file(self, path: Path) -> List[Dict]:
        """Load a memory file."""
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
