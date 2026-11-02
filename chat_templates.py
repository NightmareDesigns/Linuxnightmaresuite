import os

"""
Nightmare AI Loader v2 — Chat Templates
==========================================

Comprehensive chat template support for ALL major model families.
Auto-detects the correct template from GGUF metadata.

Supported:
- LLaMA 2/3 (Meta)
- Mistral / Mixtral
- Gemma / Gemma 2/4 (Google)
- Phi-2 / Phi-3 (Microsoft)
- Qwen / Qwen2 (Alibaba)
- DeepSeek / DeepSeek Coder
- ChatML (generic)
- Alpaca (instruction format)
- Vicuna
- Zephyr
- Command-R (Cohere)
- Custom/user-defined

Author: Built for Levi 🩸

CRITICAL FIX (v5.1):
  - Templates NO LONGER include BOS markers (<s>, <|begin_of_text|>).
    The generation loop handles BOS via tokenizer.encode(add_bos=True).
    Including BOS in the template text caused DOUBLE BOS + garbage space tokens.
  - detect_template() now checks general.name/basename and chat_template string
    BEFORE falling back to architecture. This fixes Gemma4/Mistral models that
    report architecture="llama" but need different prompt formats.
"""

from typing import Optional, List, Dict
from dataclasses import dataclass

# Debug flag — set NIGHTMARE_DEBUG=1 to enable verbose template logging
_DEBUG = os.environ.get("NIGHTMARE_DEBUG", "0") == "1"

def _dbg(msg: str):
    """Print debug message if NIGHTMARE_DEBUG=1."""
    if _DEBUG:
        print(f"[ChatTemplate DEBUG] {msg}")


@dataclass
class Message:
    """A single chat message."""
    role: str       # "system", "user", "assistant"
    content: str


class ChatTemplate:
    """Base chat template — format messages for a specific model family."""
    
    name: str = "generic"
    # Whether this template handles BOS internally (most don't — let tokenizer add it)
    handles_bos: bool = False
    
    def format_prompt(self, messages: List[Message]) -> str:
        """Format a list of messages into a model-ready prompt string."""
        raise NotImplementedError
    
    def get_stop_tokens(self) -> List[str]:
        """Return strings that signal end of generation."""
        return []
    
    def get_stop_token_ids(self) -> List[int]:
        """Return token IDs that signal end of generation (for EOS checks)."""
        return []


class LLaMA2Template(ChatTemplate):
    """Meta LLaMA 2 chat format."""
    name = "llama2"
    
    def format_prompt(self, messages: List[Message]) -> str:
        # NOTE: No <s> — BOS is added by tokenizer.encode(add_bos=True)
        prompt = ""
        system_msg = ""
        
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            elif msg.role == "user":
                if system_msg:
                    prompt += f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{msg.content} [/INST]"
                    system_msg = ""
                else:
                    prompt += f"[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                prompt += f" {msg.content} </s>"
        
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["</s>", "[INST]"]


class LLaMA3Template(ChatTemplate):
    """Meta LLaMA 3 chat format.
    
    NOTE: Special tokens like <|start_header_id|> are in the vocabulary
    for LLaMA 3 models and will be found by the greedy tokenizer.
    BOS (<|begin_of_text|>) is NOT included — handled by encode(add_bos=True).
    """
    name = "llama3"
    
    def format_prompt(self, messages: List[Message]) -> str:
        # NO <|begin_of_text|> — that IS the BOS token, added by tokenizer
        prompt = ""
        
        for msg in messages:
            prompt += f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>"
        
        # Signal assistant to respond
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["<|eot_id|>", "<|end_of_text|>"]


class MistralTemplate(ChatTemplate):
    """Mistral / Mixtral chat format.
    
    Format: [INST] message [/INST] response</s>
    BOS is added by tokenizer, NOT in template.
    """
    name = "mistral"
    
    def format_prompt(self, messages: List[Message]) -> str:
        # NOTE: No <s> — BOS is added by tokenizer.encode(add_bos=True)
        prompt = ""
        system_text = ""
        
        for msg in messages:
            if msg.role == "system":
                # Mistral prepends system to first user message
                system_text = msg.content
            elif msg.role == "user":
                if system_text:
                    prompt += f"[INST] {system_text}\n\n{msg.content} [/INST]"
                    system_text = ""
                else:
                    prompt += f"[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                prompt += f" {msg.content}</s>"
        
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["</s>"]


class GemmaTemplate(ChatTemplate):
    """Google Gemma / Gemma 2 chat format."""
    name = "gemma"

    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        system_text = ""

        for msg in messages:
            if msg.role == "system":
                if msg.content.strip():
                    system_text = msg.content.strip()
            elif msg.role == "user":
                if system_text:
                    prompt += f"<start_of_turn>user\n{system_text}\n\n{msg.content}<end_of_turn>\n"
                    system_text = ""
                else:
                    prompt += f"<start_of_turn>user\n{msg.content}<end_of_turn>\n"
            elif msg.role in ("assistant", "model"):
                prompt += f"<start_of_turn>model\n{msg.content}<end_of_turn>\n"

        if system_text:
            prompt += f"<start_of_turn>user\n{system_text}<end_of_turn>\n"

        # Signal model to respond
        prompt += "<start_of_turn>model\n"
        return prompt

    def get_stop_tokens(self) -> List[str]:
        return ["<end_of_turn>", "<eos>", "<start_of_turn>"]


class Gemma4Template(ChatTemplate):
    """Google Gemma 4 (E2B) chat format.

    Gemma 4 uses the same turn markers as Gemma 2, but the instruction-tuned
    checkpoint is sensitive to system-prompt placement.  We fold any system
    message into the first user turn with a single newline separator so the
    model never sees a bare system role.
    """
    name = "gemma4"

    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        system_text = ""

        for msg in messages:
            if msg.role == "system":
                if msg.content.strip():
                    system_text = msg.content.strip()
            elif msg.role == "user":
                if system_text:
                    prompt += f"<start_of_turn>user\n{system_text}\n{msg.content}<end_of_turn>\n"
                    system_text = ""
                else:
                    prompt += f"<start_of_turn>user\n{msg.content}<end_of_turn>\n"
            elif msg.role in ("assistant", "model"):
                prompt += f"<start_of_turn>model\n{msg.content}<end_of_turn>\n"

        if system_text:
            prompt += f"<start_of_turn>user\n{system_text}<end_of_turn>\n"

        # Signal model to respond
        prompt += "<start_of_turn>model\n"
        return prompt

    def get_stop_tokens(self) -> List[str]:
        return ["<end_of_turn>", "<eos>", "<start_of_turn>", "<turn|>"]


class Gemma4ITTemplate(Gemma4Template):
    """Gemma 4 IT (instruction-tuned) variant — identical format, separate name for logging."""
    name = "gemma4-it"


class PhiTemplate(ChatTemplate):
    """Microsoft Phi-2 / Phi-3 chat format."""
    name = "phi"
    
    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}<|end|>\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}<|end|>\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}<|end|>\n"
        
        prompt += "<|assistant|>\n"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["<|end|>", "<|endoftext|>"]


class QwenTemplate(ChatTemplate):
    """Alibaba Qwen / Qwen2 chat format (ChatML-based)."""
    name = "qwen"
    
    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        
        for msg in messages:
            prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|endoftext|>"]


class ChatMLTemplate(ChatTemplate):
    """Generic ChatML format (used by many fine-tunes)."""
    name = "chatml"
    
    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        
        for msg in messages:
            prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>"]


class DeepSeekTemplate(ChatTemplate):
    """DeepSeek / DeepSeek Coder chat format."""
    name = "deepseek"
    
    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                prompt += f"{msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"### Instruction:\n{msg.content}\n\n### Response:\n"
            elif msg.role == "assistant":
                prompt += f"{msg.content}\n\n"
        
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["### Instruction:", "<|EOT|>"]


class AlpacaTemplate(ChatTemplate):
    """Stanford Alpaca instruction format."""
    name = "alpaca"
    
    def format_prompt(self, messages: List[Message]) -> str:
        system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        instruction = ""
        
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            elif msg.role == "user":
                instruction = msg.content
        
        return f"{system}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    
    def get_stop_tokens(self) -> List[str]:
        return ["### Instruction:", "###"]


class VicunaTemplate(ChatTemplate):
    """Vicuna v1.1 chat format."""
    name = "vicuna"
    
    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                prompt += f"{msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"USER: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"ASSISTANT: {msg.content}</s>\n"
        
        prompt += "ASSISTANT:"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["</s>", "USER:"]


class ZephyrTemplate(ChatTemplate):
    """HuggingFace Zephyr chat format."""
    name = "zephyr"
    
    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}</s>\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}</s>\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}</s>\n"
        
        prompt += "<|assistant|>\n"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["</s>"]


class CommandRTemplate(ChatTemplate):
    """Cohere Command-R format."""
    name = "command-r"
    
    def format_prompt(self, messages: List[Message]) -> str:
        # NOTE: No <BOS_TOKEN> — BOS is handled by tokenizer
        prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{msg.content}<|END_OF_TURN_TOKEN|>"
            elif msg.role == "user":
                prompt += f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{msg.content}<|END_OF_TURN_TOKEN|>"
            elif msg.role == "assistant":
                prompt += f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{msg.content}<|END_OF_TURN_TOKEN|>"
        
        prompt += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["<|END_OF_TURN_TOKEN|>"]


class GenericTemplate(ChatTemplate):
    """Fallback generic format for unknown models."""
    name = "generic"
    
    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"### System:\n{msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"### User:\n{msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"### Assistant:\n{msg.content}\n\n"
        prompt += "### Assistant:\n"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["### User:", "###"]


class NightmareTemplate(ChatTemplate):
    """the assistant / Nightmare models: simple User/Assistant format."""
    name = "nightmare"
    
    def format_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                prompt += f"{msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant:"
        return prompt
    
    def get_stop_tokens(self) -> List[str]:
        return ["User:"]


# === Template Registry & Auto-Detection ===

TEMPLATES = {
    "llama2": LLaMA2Template(),
    "llama3": LLaMA3Template(),
    "llama": LLaMA3Template(),  # Default LLaMA to v3
    "llama3.1": LLaMA3Template(),
    "llama3.2": LLaMA3Template(),
    "llama3.3": LLaMA3Template(),
    "mistral": MistralTemplate(),
    "mixtral": MistralTemplate(),
    "gemma": GemmaTemplate(),
    "gemma2": GemmaTemplate(),
    "gemma3": GemmaTemplate(),
    "gemma4": Gemma4Template(),
    "gemma4-it": Gemma4ITTemplate(),
    "gemma-4": Gemma4Template(),
    "gemma_4": Gemma4Template(),
    "phi": PhiTemplate(),
    "phi2": PhiTemplate(),
    "phi3": PhiTemplate(),
    "phi3.5": PhiTemplate(),
    "phi4": PhiTemplate(),
    "qwen": QwenTemplate(),
    "qwen2": QwenTemplate(),
    "qwen2.5": QwenTemplate(),
    "qwen3": QwenTemplate(),
    "qwen35": QwenTemplate(),
    "qwen3.5": QwenTemplate(),
    "chatml": ChatMLTemplate(),
    "deepseek": DeepSeekTemplate(),
    "deepseek2": DeepSeekTemplate(),
    "deepseek3": DeepSeekTemplate(),
    "deepseek_v2": DeepSeekTemplate(),
    "deepseek_v3": DeepSeekTemplate(),
    "deepseek_r1": DeepSeekTemplate(),
    "smollm": LLaMA3Template(),
    "smollm2": ChatMLTemplate(),
    "minicpm": ChatMLTemplate(),
    "minicpm3": ChatMLTemplate(),
    "nemotron": LLaMA3Template(),
    "alpaca": AlpacaTemplate(),
    "vicuna": VicunaTemplate(),
    "zephyr": ZephyrTemplate(),
    "command-r": CommandRTemplate(),
    "generic": GenericTemplate(),
}


def detect_template(architecture: str, metadata: dict = None) -> ChatTemplate:
    """
    Auto-detect the correct chat template from model architecture/metadata.
    
    PRIORITY ORDER (critical for correctness):
      1. tokenizer.chat_template string (most reliable — model creator sets this)
      2. general.name / general.basename (catches Gemma4 reporting arch="llama")
      3. Architecture string (last resort — many models lie about this)
    
    Args:
        architecture: Model architecture string from GGUF (e.g. "gemma4", "llama", "phi3")
        metadata: Full GGUF metadata dict (for additional hints)
    
    Returns:
        Appropriate ChatTemplate instance
    """
    arch_lower = architecture.lower().strip()
    _dbg(f"detect_template called: arch='{arch_lower}', has_metadata={metadata is not None}")
    model_name = ""
    model_basename = ""
    combined = ""
    if metadata:
        model_name = str(metadata.get("general.name", "")).lower()
        model_basename = str(metadata.get("general.basename", "")).lower()
        combined = model_name + " " + model_basename
    
    # === STEP 1: Check tokenizer.chat_template (HIGHEST PRIORITY) ===
    # This is set by the model creator and is the most reliable indicator.
    if metadata:
        chat_tmpl = metadata.get("tokenizer.chat_template", "")
        if isinstance(chat_tmpl, str) and len(chat_tmpl) > 10:
            _dbg(f"  Found chat_template string ({len(chat_tmpl)} chars)")
            
            # === NIGHTMARE/TIA MODELS: only use User/the assistant if NO standard chat_template ===
            # (handled below after checking for real templates)
            
            # Check for specific format markers in the Jinja2 template
            if "<start_of_turn>" in chat_tmpl:
                # Gemma4 models often carry the same turn markers as Gemma 2.
                # Use name/basename to disambiguate before falling back to generic Gemma.
                combined_nospace = combined.replace(" ", "").replace("-", "").replace("_", "")
                if "gemma4" in combined_nospace or "gemma-4" in combined:
                    _dbg(f"  → Detected GEMMA4 from chat_template + model name")
                    return TEMPLATES["gemma4"]
                _dbg(f"  → Detected GEMMA from chat_template")
                return TEMPLATES["gemma"]
            elif "start_header_id" in chat_tmpl:
                _dbg(f"  → Detected LLAMA3 from chat_template")
                return TEMPLATES["llama3"]
            elif "[INST]" in chat_tmpl:
                # Could be Mistral or LLaMA 2 — check for <<SYS>>
                if "<<SYS>>" in chat_tmpl:
                    _dbg(f"  → Detected LLAMA2 from chat_template")
                    return TEMPLATES["llama2"]
                else:
                    _dbg(f"  → Detected MISTRAL from chat_template")
                    return TEMPLATES["mistral"]
            elif "im_start" in chat_tmpl:
                # Could be Qwen or generic ChatML
                if "qwen" in chat_tmpl.lower():
                    _dbg(f"  → Detected QWEN from chat_template")
                    return TEMPLATES["qwen"]
                _dbg(f"  → Detected CHATML from chat_template")
                return TEMPLATES["chatml"]
            elif "<|system|>" in chat_tmpl or "<|user|>" in chat_tmpl:
                # These markers are used by several families.
                # Only treat as Phi when explicit <|end|> separators are present.
                ct_lower = chat_tmpl.lower()
                if "<|end|>" in ct_lower:
                    _dbg(f"  → Detected PHI from chat_template (<|end|> markers)")
                    return TEMPLATES["phi"]
                if "zephyr" in ct_lower or "tinyllama" in combined or "eos_token" in ct_lower:
                    _dbg(f"  → Detected ZEPHYR from chat_template")
                    return TEMPLATES["zephyr"]
                _dbg(f"  → Detected CHATML-like user/assistant format from chat_template")
                return TEMPLATES["zephyr"]
            elif "### Instruction" in chat_tmpl:
                _dbg(f"  → Detected DEEPSEEK from chat_template")
                return TEMPLATES["deepseek"]
    
    # === STEP 2: Check general.name and general.basename ===
    # This catches models that report wrong architecture (e.g., Gemma4 as "llama")
    if metadata:
        _dbg(f"  Model name='{model_name}', basename='{model_basename}'")
        
        combined_nospace = combined.replace(" ", "").replace("-", "").replace("_", "")
        if "gemma4" in combined_nospace:
            _dbg(f"  → Detected GEMMA4 from model name/basename")
            return TEMPLATES["gemma4"]
        elif "gemma2" in combined_nospace:
            _dbg(f"  → Detected GEMMA2 from model name/basename")
            return TEMPLATES["gemma"]
        elif "gemma" in combined:
            _dbg(f"  → Detected GEMMA from model name/basename")
            return TEMPLATES["gemma"]
        elif "tinyllama" in combined:
            _dbg(f"  → Detected TINYLLAMA from model name/basename")
            return TEMPLATES["zephyr"]
        elif "mistral" in combined or "mixtral" in combined:
            _dbg(f"  → Detected MISTRAL from model name/basename")
            return TEMPLATES["mistral"]
        elif "qwen" in combined:
            _dbg(f"  → Detected QWEN from model name/basename")
            return TEMPLATES["qwen"]
        elif "phi" in combined:
            _dbg(f"  → Detected PHI from model name/basename")
            return TEMPLATES["phi"]
        elif "deepseek" in combined:
            _dbg(f"  → Detected DEEPSEEK from model name/basename")
            return TEMPLATES["deepseek"]
        elif "zephyr" in combined:
            _dbg(f"  → Detected ZEPHYR from model name/basename")
            return TEMPLATES["zephyr"]
        elif "command" in combined and "cohere" in combined:
            _dbg(f"  → Detected COMMAND-R from model name/basename")
            return TEMPLATES["command-r"]
        elif "vicuna" in combined:
            _dbg(f"  → Detected VICUNA from model name/basename")
            return TEMPLATES["vicuna"]
        elif "tia" in combined or "nightmare" in combined:
            _dbg(f"  → Detected NIGHTMARE/TIA from model name/basename")
            return NightmareTemplate()
        elif "llama" in combined:
            # Check if it's LLaMA 2 or 3
            if "llama-2" in combined or "llama2" in combined:
                _dbg(f"  → Detected LLAMA2 from model name/basename")
                return TEMPLATES["llama2"]
            else:
                _dbg(f"  → Detected LLAMA3 from model name/basename")
                return TEMPLATES["llama3"]

    # === STEP 3: Fall back to architecture string ===
    _dbg(f"  Falling back to architecture match: '{arch_lower}'")
    
    # Direct match
    if arch_lower in TEMPLATES:
        _dbg(f"  → Direct arch match: {arch_lower}")
        return TEMPLATES[arch_lower]

    # Gemma 4 may report itself as plain "gemma" in GGUF metadata — catch it
    # by checking for Gemma 4 specific keys before falling back to generic Gemma.
    if metadata and arch_lower == "gemma":
        if any(k in str(metadata.keys()).lower() for k in ("gemma4", "gemma-4")):
            _dbg(f"  → Detected GEMMA4 from metadata keys")
            return TEMPLATES["gemma4"]
        if metadata.get("gemma.attention.key_length") == 512:
            # Gemma 4 E2B uses key_length=512 (Gemma 2 uses 256)
            _dbg(f"  → Detected GEMMA4 from key_length=512")
            return TEMPLATES["gemma4"]
    
    # Fuzzy matching
    for key in TEMPLATES:
        if key in arch_lower or arch_lower in key:
            _dbg(f"  → Fuzzy arch match: {arch_lower} ~ {key}")
            return TEMPLATES[key]
    
    # === Fallback ===
    # Nightmare/Assistant models: ONLY use User/Assistant format as LAST RESORT
    # (i.e., when no standard template, name, or architecture matched)
    if metadata:
        if "nightmare" in str(metadata.get("nightmare.architecture", "")).lower():
            _dbg(f"  → Fallback: Nightmare model — using User/Assistant format")
            return NightmareTemplate()
        if "tia" in str(metadata.get("general.name", "")).lower():
            _dbg(f"  → Fallback: Assistant model — using User/Assistant format")
            return NightmareTemplate()
    
    _dbg(f"  → No match found, using GENERIC template")
    return TEMPLATES["generic"]


def format_chat(
    architecture: str,
    system_prompt: str = "",
    user_message: str = "",
    history: List[Dict[str, str]] = None,
    metadata: dict = None
) -> str:
    """
    Convenience function: format a single chat turn.
    
    Args:
        architecture: Model architecture
        system_prompt: System message
        user_message: User's message
        history: Previous messages as [{"role": "user/assistant", "content": "..."}]
        metadata: GGUF metadata for template detection
    
    Returns:
        Formatted prompt string ready for tokenization.
        NOTE: Does NOT include BOS token — caller must use encode(add_bos=True).
    """
    template = detect_template(architecture, metadata)
    _dbg(f"format_chat: using template '{template.name}' for arch='{architecture}'")
    
    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    
    if history:
        for h in history:
            messages.append(Message(role=h["role"], content=h["content"]))
        if user_message and (history[-1].get("content") != user_message or history[-1].get("role") != "user"):
            messages.append(Message(role="user", content=user_message))
    elif user_message:
        messages.append(Message(role="user", content=user_message))
    
    prompt = template.format_prompt(messages)
    _dbg(f"format_chat: prompt length = {len(prompt)} chars")
    _dbg(f"format_chat: first 200 chars = {repr(prompt[:200])}")
    
    return prompt


def get_stop_strings(architecture: str, metadata: dict = None) -> List[str]:
    """Get stop strings for the detected template."""
    template = detect_template(architecture, metadata)
    return template.get_stop_tokens()


def get_template_name(architecture: str, metadata: dict = None) -> str:
    """Get the name of the detected template (for display/debugging)."""
    template = detect_template(architecture, metadata)
    return template.name
