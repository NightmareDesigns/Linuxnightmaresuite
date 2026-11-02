#!/usr/bin/env python3
"""
GGUF Model Loader
==================
Handles discovery, loading, and metadata extraction for GGUF models.
Used by the Nightmare Suite to discover large models (>= 26B) efficiently.
"""

import os
from pathlib import Path


class GGUFModelLoader:
    """
    Loader for GGUF model files that provides metadata discovery.
    
    This class is used for loading large models (>= 26GB) where we need
    to extract comprehensive metadata without fully loading the model weights.
    """
    
    def __init__(self):
        self._model_paths = {}
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a GGUF model and return True if successful.
        
        Args:
            filepath: Path to the .gguf file
            
        Returns:
            True if loading succeeded, False otherwise
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return False
        
        try:
            # Use the suite's GGUF parser for real metadata extraction
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from gguf_parser import GGUFParser
            parser = GGUFParser(str(filepath))
            model = parser.parse()
            arch = model.metadata.get('general.architecture', 'unknown')
            dim = model.metadata.get(f'{arch}.embedding_length', 0)
            layers = model.metadata.get(f'{arch}.block_count', 0)
            self._model_paths[str(filepath)] = {
                'architecture': arch,
                'dim': dim,
                'n_layers': layers,
                'vocab_size': model.metadata.get(f'{arch}.vocabulary_length', 0),
                'n_tensors': len(model.tensors),
            }
            print(f"[GGUFModelLoader] {filepath.name}: {arch} {dim}d x {layers}L, {len(model.tensors)} tensors")
            parser.close()
            return True
            
        except Exception as e:
            print(f"[GGUFModelLoader] Error loading {filepath}: {e}")
            return False
    
    def get_model_info(self, filepath: str) -> dict:
        """Get metadata about a GGUF model."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return None
        
        try:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            return {
                "path": str(filepath),
                "size_mb": round(size_mb, 1),
                "name": filepath.stem
            }
        except Exception as e:
            print(f"[GGUFModelLoader] Error getting info for {filepath}: {e}")
            return None
    
    def discover_models(self, directory: str) -> list:
        """Discover all GGUF models in a directory."""
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        gguf_files = sorted([str(f) for f in directory.rglob("*.gguf")])
        
        results = []
        for filepath in gguf_files:
            info = self.get_model_info(filepath)
            if info:
                results.append(info)
        
        return results
