"""Text normalization utilities"""

import re


class TextNormalizer:
    """Normalize text for consistent processing"""
    
    def normalize(self, text: str) -> str:
        """Apply basic normalization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text