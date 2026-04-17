"""Advanced chat interface with COH integration"""

from .chat_manager import COHDialogueManager


class GISMOLChat(COHDialogueManager):
    """Advanced chat interface with hierarchical context management"""
    
    def set_focus_object(self, obj_name: str) -> None:
        self.repository.set_focus_object(obj_name)
    
    def respond(self, user_input: str) -> str:
        # Enhanced response with context
        focus = self.repository.focus_object
        if focus:
            return f"{focus.name}: {super().respond(user_input)}"
        return super().respond(user_input)