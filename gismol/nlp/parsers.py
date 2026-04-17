"""Text parsing utilities"""


class TextParser:
    """Parse text into structured components"""
    
    def parse(self, text: str) -> dict:
        """Basic parsing: split into words"""
        return {
            'words': text.split(),
            'length': len(text),
            'sentences': text.split('.')
        }