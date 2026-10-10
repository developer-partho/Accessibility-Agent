"""
Keyboard Mapper
Maps voice commands to keyboard actions and key presses
Handles misspelling correction and fuzzy matching
"""

import logging
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher
import re

logger = logging.getLogger("KeyboardMapper")


class KeyboardMapper:
    """Maps voice input to keyboard actions"""
    
    def __init__(self):
        """Initialize keyboard mapper with all mappings"""
        self.shortcuts_map = self._initialize_shortcuts()
        self.key_names_map = self._initialize_key_names()
        self.command_patterns = self._initialize_command_patterns()
        self.misspelling_corrections = self._initialize_corrections()
    
    def _initialize_shortcuts(self) -> Dict[str, str]:
        """Initialize common keyboard shortcuts"""
        return {
            # Navigation
            'new chat': 'ctrl+f',
            'new': 'ctrl+n',
            'open': 'ctrl+o',
            'save': 'ctrl+s',
            'save as': 'ctrl+shift+s',
            'close': 'ctrl+w',
            'close window': 'ctrl+w',
            'exit': 'alt+f4',
            
            # Editing
            'undo': 'ctrl+z',
            'redo': 'ctrl+y',
            'cut': 'ctrl+x',
            'copy': 'ctrl+c',
            'paste': 'ctrl+v',
            'select all': 'ctrl+a',
            'find': 'ctrl+f',
            'replace': 'ctrl+h',
            'delete': 'delete',
            'backspace': 'backspace',
            
            # Formatting
            'bold': 'ctrl+b',
            'italic': 'ctrl+i',
            'underline': 'ctrl+u',
            'strikethrough': 'ctrl+shift+x',
            
            # Application
            'print': 'ctrl+p',
            'help': 'f1',
            'refresh': 'f5',
            'zoom in': 'ctrl+plus',
            'zoom out': 'ctrl+minus',
            'zoom': 'ctrl+0',
            
            # Window management
            'minimize': 'alt+f9',
            'maximize': 'alt+f10',
            'restore': 'alt+f5',
            'fullscreen': 'f11',
            
            # Navigation
            'home': 'home',
            'end': 'end',
            'page up': 'pageup',
            'page down': 'pagedown',
            'go to start': 'ctrl+home',
            'go to end': 'ctrl+end',
            
            # Tab/window switching
            'next tab': 'ctrl+tab',
            'previous tab': 'ctrl+shift+tab',
            'next window': 'alt+tab',
            'previous window': 'alt+shift+tab',
            
            # Browser specific
            'back': 'alt+left',
            'forward': 'alt+right',
            'reload': 'f5',
            'new tab': 'ctrl+t',
            'close tab': 'ctrl+w',
            'reopen tab': 'ctrl+shift+t',
            
            # Chat/Messaging
            'send': 'enter',
            'new message': 'ctrl+n',
            'reply': 'ctrl+r',
            'forward': 'ctrl+shift+f',
            
            # Special
            'enter': 'enter',
            'return': 'enter',
            'space': 'space',
            'tab': 'tab',
            'escape': 'escape',
            'delete': 'delete',
            'backspace': 'backspace',
        }
    
    def _initialize_key_names(self) -> Dict[str, str]:
        """Initialize all individual key names"""
        keys = {}
        
        # Letters A-Z
        for i, letter in enumerate('abcdefghijklmnopqrstuvwxyz'):
            keys[letter] = letter
            keys[letter.upper()] = letter.upper()
        
        # Numbers 0-9
        for i in range(10):
            keys[str(i)] = str(i)
        
        # Special keys
        special = {
            'enter': 'enter',
            'return': 'enter',
            'tab': 'tab',
            'space': 'space',
            'spacebar': 'space',
            'escape': 'escape',
            'esc': 'escape',
            'delete': 'delete',
            'backspace': 'backspace',
            'back': 'backspace',
            'home': 'home',
            'end': 'end',
            'pageup': 'pageup',
            'page up': 'pageup',
            'pagedown': 'pagedown',
            'page down': 'pagedown',
            'insert': 'insert',
            'arrow up': 'up',
            'up': 'up',
            'arrow down': 'down',
            'down': 'down',
            'arrow left': 'left',
            'left': 'left',
            'arrow right': 'right',
            'right': 'right',
            
            'shift': 'shift',
            'control': 'ctrl',
            'ctrl': 'ctrl',
            'alt': 'alt',
            'command': 'cmd',
            'windows': 'win',
            'win': 'win',
            
            'f1': 'f1',
            'f2': 'f2',
            'f3': 'f3',
            'f4': 'f4',
            'f5': 'f5',
            'f6': 'f6',
            'f7': 'f7',
            'f8': 'f8',
            'f9': 'f9',
            'f10': 'f10',
            'f11': 'f11',
            'f12': 'f12',
            
            'plus': '+',
            'minus': '-',
            'equals': '=',
            'dash': '-',
            'period': '.',
            'dot': '.',
            'comma': ',',
            'semicolon': ';',
            'colon': ':',
            'quote': "'",
            'double quote': '"',
            'slash': '/',
            'backslash': '\\',
            'question mark': '?',
            'exclamation': '!',
            'at': '@',
            'hash': '#',
            'dollar': '$',
            'percent': '%',
            'ampersand': '&',
            'asterisk': '*',
            'parenthesis': '(',
            'bracket': '[',
            'brace': '{',
        }
        
        keys.update(special)
        return keys
    
    def _initialize_command_patterns(self) -> List[Dict]:
        """Initialize regex patterns for command recognition"""
        return [
            # Keyboard shortcut patterns
            {
                'pattern': r'(ctrl|control)\s*\+?\s*([a-z])',
                'type': 'shortcut',
                'format': lambda m: f"ctrl+{m.group(2).lower()}"
            },
            {
                'pattern': r'(shift)\s*\+?\s*([a-z])',
                'type': 'shortcut',
                'format': lambda m: f"shift+{m.group(2).lower()}"
            },
            {
                'pattern': r'(alt)\s*\+?\s*([a-z])',
                'type': 'shortcut',
                'format': lambda m: f"alt+{m.group(2).lower()}"
            },
            {
                'pattern': r'(ctrl|control)\s*\+?\s*(shift)\s*\+?\s*([a-z])',
                'type': 'shortcut',
                'format': lambda m: f"ctrl+shift+{m.group(3).lower()}"
            },
            
            # Type/write patterns
            {
                'pattern': r'^(?:type|write|say|text)\s+(.+)',
                'type': 'text',
                'format': lambda m: m.group(1)
            },
            
            # Single character
            {
                'pattern': r'^(?:press\s+)?(?:the\s+)?character\s+([a-z0-9])',
                'type': 'key',
                'format': lambda m: m.group(1)
            },
        ]
    
    def _initialize_corrections(self) -> Dict[str, str]:
        """Initialize common misspelling corrections"""
        return {
            # Common voice recognition mistakes
            'controle': 'ctrl',
            'controll': 'ctrl',
            'cntrl': 'ctrl',
            'tab': 'tab',
            'tabs': 'tab',
            'escape': 'escape',
            'escap': 'escape',
            'spce': 'space',
            'space': 'space',
            'delite': 'delete',
            'delte': 'delete',
            'entrer': 'enter',
            'entr': 'enter',
            'sav': 'save',
            'savee': 'save',
            'opne': 'open',
            'clos': 'close',
            'closse': 'close',
            'find': 'find',
            'replac': 'replace',
            'undo': 'undo',
            'redo': 'redo',
            'cop': 'copy',
            'coppy': 'copy',
            'past': 'paste',
            'paste': 'paste',
            'cut': 'cut',
            'nex': 'next',
            'prev': 'prev',
            'previus': 'previous',
        }
    
    def map_voice_to_keyboard(self, voice_input: str) -> Dict:
        """
        Map voice input to keyboard action
        
        Args:
            voice_input: Voice command or text
        
        Returns:
            Dictionary with action type and key/text to execute
        """
        voice_input = voice_input.lower().strip()
        
        # Step 1: Apply misspelling corrections
        corrected_input = self._apply_corrections(voice_input)
        
        # Step 2: Check for exact shortcut match
        if corrected_input in self.shortcuts_map:
            return {
                'action': 'shortcut',
                'keys': self.shortcuts_map[corrected_input],
                'original_input': voice_input,
                'corrected_input': corrected_input,
                'confidence': 1.0,
            }
        
        # Step 3: Check for fuzzy shortcut match
        fuzzy_match = self._fuzzy_match_shortcut(corrected_input)
        if fuzzy_match:
            return {
                'action': 'shortcut',
                'keys': fuzzy_match['keys'],
                'original_input': voice_input,
                'corrected_input': corrected_input,
                'confidence': fuzzy_match['confidence'],
            }
        
        # Step 4: Check for individual key
        if corrected_input in self.key_names_map:
            return {
                'action': 'key',
                'key': self.key_names_map[corrected_input],
                'original_input': voice_input,
                'confidence': 1.0,
            }
        
        # Step 5: Check for pattern matches
        for pattern_info in self.command_patterns:
            pattern = pattern_info['pattern']
            match = re.match(pattern, voice_input, re.IGNORECASE)
            if match:
                result_value = pattern_info['format'](match)
                
                if pattern_info['type'] == 'text':
                    return {
                        'action': 'type',
                        'text': result_value,
                        'original_input': voice_input,
                        'confidence': 0.95,
                    }
                elif pattern_info['type'] == 'shortcut':
                    return {
                        'action': 'shortcut',
                        'keys': result_value,
                        'original_input': voice_input,
                        'confidence': 0.9,
                    }
                elif pattern_info['type'] == 'key':
                    return {
                        'action': 'key',
                        'key': result_value,
                        'original_input': voice_input,
                        'confidence': 0.9,
                    }
        
        # Step 6: Assume it's text to type if nothing matched
        return {
            'action': 'type',
            'text': voice_input,
            'original_input': voice_input,
            'confidence': 0.5,  # Low confidence - might need confirmation
        }
    
    def _apply_corrections(self, text: str) -> str:
        """Apply misspelling corrections"""
        words = text.split()
        corrected = []
        
        for word in words:
            if word in self.misspelling_corrections:
                corrected.append(self.misspelling_corrections[word])
            else:
                corrected.append(word)
        
        return ' '.join(corrected)
    
    def _fuzzy_match_shortcut(self, input_text: str, threshold: float = 0.75) -> Optional[Dict]:
        """Fuzzy match against shortcuts"""
        best_match = None
        best_score = threshold
        
        for shortcut_name, shortcut_keys in self.shortcuts_map.items():
            score = SequenceMatcher(None, input_text, shortcut_name).ratio()
            
            if score > best_score:
                best_score = score
                best_match = {
                    'keys': shortcut_keys,
                    'confidence': score,
                }
        
        return best_match
    
    def parse_modifier_keys(self, keys_str: str) -> List[str]:
        """
        Parse modifier keys from shortcut string
        
        Args:
            keys_str: Shortcut string like "ctrl+shift+s"
        
        Returns:
            List of parsed modifiers and keys
        """
        parts = keys_str.lower().split('+')
        return [p.strip() for p in parts if p.strip()]
    
    def is_shortcut(self, keys_str: str) -> bool:
        """Check if input is a keyboard shortcut"""
        return any(mod in keys_str.lower() for mod in ['ctrl', 'shift', 'alt'])
    
    def is_text_input(self, action_type: str) -> bool:
        """Check if action is text input"""
        return action_type == 'type'
    
    def recognize_modifier_pattern(self, voice_input: str) -> Optional[Dict]:
        """Recognize patterns like 'control f' or 'shift tab'"""
        # Pattern: [modifier(s)] + [key]
        pattern = r'^(?:(ctrl|shift|alt|control|command|windows|win)\s*\+?\s*)*([a-z0-9]|f[0-9]+|tab|enter|space|escape|delete|backspace)$'
        
        match = re.match(pattern, voice_input.lower())
        if match:
            return {
                'modifiers': match.group(1),
                'key': match.group(2),
                'confidence': 0.95,
            }
        
        return None
    
    def get_all_shortcuts(self) -> Dict[str, str]:
        """Get all available shortcuts"""
        return self.shortcuts_map.copy()
    
    def get_all_keys(self) -> Dict[str, str]:
        """Get all available keys"""
        return self.key_names_map.copy()
    
    def add_custom_shortcut(self, voice_command: str, keyboard_shortcut: str):
        """Add custom shortcut mapping"""
        self.shortcuts_map[voice_command.lower()] = keyboard_shortcut.lower()
        logger.info(f"Custom shortcut added: '{voice_command}' → '{keyboard_shortcut}'")
    
    def add_custom_correction(self, misspelled: str, correct: str):
        """Add custom misspelling correction"""
        self.misspelling_corrections[misspelled.lower()] = correct.lower()
        logger.info(f"Custom correction added: '{misspelled}' → '{correct}'")


def create_keyboard_mapper() -> KeyboardMapper:
    """Factory function to create keyboard mapper"""
    return KeyboardMapper()
