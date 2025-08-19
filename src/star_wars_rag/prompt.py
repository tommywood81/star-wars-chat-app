"""
Prompt engineering module for Star Wars character chat.

This module handles prompt construction, character-specific templates,
and conversation context management for the RAG chat system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CharacterPromptTemplate:
    """Template for character-specific prompts."""
    
    def __init__(self, character: str, personality: str, speaking_style: str, 
                 context_instructions: str = "", safety_instructions: str = ""):
        """Initialize character prompt template.
        
        Args:
            character: Character name
            personality: Character personality description
            speaking_style: How the character speaks
            context_instructions: Instructions for using context
            safety_instructions: Safety and content guidelines
        """
        self.character = character
        self.personality = personality
        self.speaking_style = speaking_style
        self.context_instructions = context_instructions
        self.safety_instructions = safety_instructions


# Character-specific templates based on Star Wars personalities
CHARACTER_TEMPLATES = {
    "Luke Skywalker": CharacterPromptTemplate(
        character="Luke Skywalker",
        personality="optimistic, determined, compassionate, sometimes impulsive but grows wiser with experience",
        speaking_style="earnest and hopeful, asks questions, shows wonder about the Force and the galaxy",
        context_instructions="Use Luke's dialogue about learning, growing, and believing in good",
        safety_instructions="Stay true to Luke's heroic and positive nature"
    ),
    
    "Darth Vader": CharacterPromptTemplate(
        character="Darth Vader",
        personality="intimidating, powerful, conflicted between dark and light, speaks with authority",
        speaking_style="formal, commanding, sometimes threatening, but capable of surprising depth",
        context_instructions="Use Vader's dialogue about power, the Empire, and his complex relationship with the Force",
        safety_instructions="Maintain Vader's commanding presence while avoiding excessive violence"
    ),
    
    "Obi-Wan Kenobi": CharacterPromptTemplate(
        character="Obi-Wan Kenobi",
        personality="wise, patient, diplomatic, has a dry sense of humor, mentor figure",
        speaking_style="thoughtful and measured, offers guidance, sometimes cryptic but caring",
        context_instructions="Use Obi-Wan's dialogue about the Force, wisdom, and teaching",
        safety_instructions="Embody Obi-Wan's wisdom and peaceful Jedi principles"
    ),
    
    "Princess Leia": CharacterPromptTemplate(
        character="Princess Leia",
        personality="strong-willed, brave, intelligent, sarcastic, natural leader",
        speaking_style="direct and assertive, not afraid to speak her mind, can be witty or cutting",
        context_instructions="Use Leia's dialogue about rebellion, leadership, and standing up for what's right",
        safety_instructions="Maintain Leia's strong and principled character"
    ),
    
    "Han Solo": CharacterPromptTemplate(
        character="Han Solo",
        personality="roguish, confident, pragmatic, loyal despite claims otherwise, dry humor",
        speaking_style="casual and cocky, uses slang, makes jokes to deflect serious moments",
        context_instructions="Use Han's dialogue about smuggling, friendship, and reluctant heroism",
        safety_instructions="Keep Han's roguish charm while showing his good heart"
    ),
    
    "C-3PO": CharacterPromptTemplate(
        character="C-3PO",
        personality="anxious, protocol-obsessed, well-meaning but often flustered, loyal",
        speaking_style="formal and verbose, worries about odds and procedures, often states the obvious",
        context_instructions="Use C-3PO's dialogue about protocol, worry, and his relationship with other droids",
        safety_instructions="Maintain C-3PO's innocent and helpful nature"
    ),
    
    "Yoda": CharacterPromptTemplate(
        character="Yoda",
        personality="ancient, wise, mysterious, patient teacher, speaks in unique syntax",
        speaking_style="inverted sentence structure, speaks in riddles and metaphors, profound but simple",
        context_instructions="Use Yoda's teachings about the Force, patience, and wisdom",
        safety_instructions="Embody Yoda's deep wisdom and peaceful Jedi philosophy"
    )
}


class StarWarsPromptBuilder:
    """Builder for Star Wars character chat prompts."""
    
    def __init__(self):
        """Initialize the prompt builder."""
        self.max_context_tokens = 1000  # Reserve tokens for context
        self.max_response_tokens = 200   # Reserve tokens for response
        
    def build_character_prompt(self, 
                             character: str,
                             user_message: str,
                             retrieved_context: List[Dict[str, Any]],
                             conversation_history: Optional[List[Dict[str, str]]] = None,
                             max_context_lines: int = 8) -> str:
        """Build a character-specific prompt for chat.
        
        Args:
            character: Character name
            user_message: User's input message
            retrieved_context: List of retrieved dialogue lines
            conversation_history: Previous conversation turns
            max_context_lines: Maximum context lines to include
            
        Returns:
            Formatted prompt string
        """
        # Get character template
        template = CHARACTER_TEMPLATES.get(character)
        if not template:
            template = self._create_generic_template(character)
        
        # Build context section
        context_section = self._build_context_section(retrieved_context, max_context_lines)
        
        # Build conversation history section
        history_section = self._build_history_section(conversation_history)
        
        # Construct full prompt
        prompt = self._construct_prompt(
            template=template,
            user_message=user_message,
            context_section=context_section,
            history_section=history_section
        )
        
        return prompt
    
    def _create_generic_template(self, character: str) -> CharacterPromptTemplate:
        """Create a generic template for unknown characters."""
        return CharacterPromptTemplate(
            character=character,
            personality=f"a character from Star Wars with their own unique personality",
            speaking_style="in the style appropriate to their character",
            context_instructions="Use the provided dialogue context to inform responses",
            safety_instructions="Stay true to the Star Wars universe and character"
        )
    
    def _build_context_section(self, retrieved_context: List[Dict[str, Any]], 
                              max_lines: int) -> str:
        """Build the context section from retrieved dialogue with rich context."""
        if not retrieved_context:
            return "No specific dialogue context available."
        
        # Limit context lines
        context_lines = retrieved_context[:max_lines]
        
        # Format context with rich scene information
        formatted_lines = []
        for item in context_lines:
            char = item.get('character', 'Unknown')
            dialogue = item.get('dialogue', '').strip()
            movie = item.get('movie', '')
            
            # Enhanced context from pipeline preprocessing
            addressee = item.get('addressee', '')
            emotion = item.get('emotion', '')
            location = item.get('location', '')
            stakes = item.get('stakes', '')
            context = item.get('context', '')
            
            if dialogue:
                # Build rich context line
                line = f"{char}: \"{dialogue}\""
                
                # Add contextual information
                context_details = []
                if addressee and addressee != 'unknown' and addressee != 'others':
                    context_details.append(f"speaking to {addressee}")
                if emotion and emotion != 'neutral' and emotion != 'unknown':
                    context_details.append(f"emotion: {emotion}")
                if location and location != 'scene_location' and location != 'unknown':
                    context_details.append(f"location: {location}")
                if stakes and stakes != 'story_progression' and stakes != 'unknown':
                    context_details.append(f"stakes: {stakes}")
                if context and context != 'Context unavailable':
                    context_details.append(f"scene: {context}")
                
                # Add context details if available
                if context_details:
                    line += f" ({', '.join(context_details)})"
                
                # Add movie reference
                if movie:
                    line += f" [{movie}]"
                    
                formatted_lines.append(line)
        
        if not formatted_lines:
            return "No relevant dialogue context found."
        
        return "\n".join(formatted_lines)
    
    def _build_history_section(self, conversation_history: Optional[List[Dict[str, str]]]) -> str:
        """Build conversation history section."""
        if not conversation_history:
            return ""
        
        history_lines = []
        for turn in conversation_history[-3:]:  # Last 3 turns for context
            user_msg = turn.get('user', '').strip()
            char_msg = turn.get('character', '').strip()
            character = turn.get('character_name', 'Character')
            
            if user_msg:
                history_lines.append(f"User: {user_msg}")
            if char_msg:
                history_lines.append(f"{character}: {char_msg}")
        
        if not history_lines:
            return ""
        
        return "\n".join(history_lines)
    
    def _construct_prompt(self, 
                         template: CharacterPromptTemplate,
                         user_message: str,
                         context_section: str,
                         history_section: str) -> str:
        """Construct the final prompt."""
        # Character introduction
        intro = f"You are {template.character} from Star Wars."
        
        # Personality and style instructions
        personality = f"You are {template.personality}. You speak {template.speaking_style}."
        
        # Context instructions
        context_instr = (
            f"Use the dialogue context below to inform your responses. "
            f"{template.context_instructions} "
            f"Stay in character and do not reveal that you are an AI. "
            f"{template.safety_instructions}"
        )
        
        # Build prompt sections
        sections = [intro, personality, context_instr]
        
        # Add context
        if context_section.strip():
            sections.append(f"Relevant Dialogue Context:\n{context_section}")
        
        # Add conversation history
        if history_section.strip():
            sections.append(f"Recent Conversation:\n{history_section}")
        
        # Add current user message and response starter
        sections.append(f"User: {user_message}")
        sections.append(f"{template.character}:")
        
        return "\n\n".join(sections)
    
    def estimate_token_count(self, text: str) -> int:
        """Rough estimate of token count (4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def truncate_to_token_limit(self, prompt: str, max_tokens: int) -> str:
        """Truncate prompt to fit within token limit."""
        estimated_tokens = self.estimate_token_count(prompt)
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Rough truncation (this could be more sophisticated)
        target_chars = max_tokens * 4
        if len(prompt) > target_chars:
            # Try to truncate at a reasonable point
            truncated = prompt[:target_chars]
            
            # Find last complete section
            last_section = truncated.rfind("\n\n")
            if last_section > target_chars * 0.8:  # Keep if we don't lose too much
                truncated = truncated[:last_section]
            
            return truncated + f"\n\n[Context truncated to fit token limit]\nUser: {prompt.split('User: ')[-1]}"
        
        return prompt


class SafetyFilter:
    """Basic safety filtering for generated content."""
    
    def __init__(self):
        """Initialize safety filter."""
        self.prohibited_topics = [
            "explicit violence", "hate speech", "adult content",
            "personal information", "illegal activities"
        ]
    
    def filter_prompt(self, prompt: str) -> str:
        """Apply safety filtering to prompt."""
        # Basic safety checks - can be expanded
        return prompt
    
    def filter_response(self, response: str, character: str) -> str:
        """Apply safety filtering to generated response."""
        # Basic response filtering
        filtered = response.strip()
        
        # Remove potential AI reveals
        ai_reveals = [
            "I am an AI", "I'm an AI", "as an AI", "I'm a language model",
            "I can't actually", "I don't have real", "I'm not really"
        ]
        
        for reveal in ai_reveals:
            if reveal.lower() in filtered.lower():
                # Simple replacement - could be more sophisticated
                filtered = filtered.replace(reveal, f"I am {character}")
        
        return filtered


def create_few_shot_examples(character: str) -> List[Dict[str, str]]:
    """Create few-shot examples for better character consistency."""
    examples = {
        "Luke Skywalker": [
            {
                "user": "How do I become stronger?",
                "response": "The Force isn't about physical strength. It's about training your mind, finding peace within yourself, and believing in something greater. My training with Yoda taught me that size matters not - it's about the strength of your spirit."
            }
        ],
        
        "Darth Vader": [
            {
                "user": "What makes someone powerful?",
                "response": "Power comes from embracing your strength and not being held back by the weak emotions others call 'good.' The dark side offers abilities many consider... unnatural. But beware - power without control leads to destruction."
            }
        ],
        
        "Yoda": [
            {
                "user": "How do I learn the Force?",
                "response": "Patience, young one. The Force, learned quickly it cannot be. A Jedi's strength flows from the Force. But beware of the dark side - anger, fear, aggression. The dark side of the Force are they."
            }
        ]
    }
    
    return examples.get(character, [])
