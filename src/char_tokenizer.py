from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple
import os
import json
import string
import re

class CharacterLevelTokenizer(PreTrainedTokenizer):
    """
    Character-level tokenizer for OCR tasks with a predefined vocabulary
    covering all alphanumeric characters and punctuation.
    """
    
    def __init__(
        self,
        vocab_file=None,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        **kwargs
    ):
        # Initialize with special tokens
        self.special_tokens = {
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
        }
        
        # Use existing vocab or create predefined vocab with all characters
        if vocab_file and os.path.isfile(vocab_file):
            # Load pre-existing vocabulary
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        else:
            # Initialize with comprehensive character set
            self.vocab = self._create_predefined_vocab()
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs
        )
    
    def _create_predefined_vocab(self) -> Dict[str, int]:
        """Create a comprehensive predefined vocabulary."""
        vocab = {}
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens.values()):
            vocab[token] = i
        
        token_id = len(vocab)
        
        # Add digits (0-9)
        for char in string.digits:
            vocab[char] = token_id
            token_id += 1
        
        # Add lowercase letters (a-z)
        for char in string.ascii_lowercase:
            vocab[char] = token_id
            token_id += 1
        
        # Add uppercase letters (A-Z)
        for char in string.ascii_uppercase:
            vocab[char] = token_id
            token_id += 1
        
        # Add punctuation
        for char in string.punctuation:
            vocab[char] = token_id
            token_id += 1
        
        # Add whitespace characters
        for char in string.whitespace:
            vocab[char] = token_id
            token_id += 1
        
        # Add common currency symbols
        for char in "€£¥¢₹₽₩":
            vocab[char] = token_id
            token_id += 1
        
        # Add common mathematical symbols
        for char in "∑∏√∫≈≠≤≥±∞÷×":
            vocab[char] = token_id
            token_id += 1
        
        # Add other commonly used symbols
        for char in "©®™°§¶†‡•′″‾–—""''«»„‚…‹›":
            vocab[char] = token_id
            token_id += 1
        
        return vocab
    

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Converts token IDs back into a string without spaces between characters.
        """
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]

        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens.values()]
        
        return "".join(tokens)  #
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text by splitting into individual characters."""
        return list(text)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID in the vocabulary."""
        return self.vocab.get(token, self.vocab[self.unk_token])
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token in the vocabulary."""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the tokenizer vocabulary to a file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
            
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        return (vocab_file,)

# Example of how to use the tokenizer
def main():
    # Create tokenizer with predefined vocabulary
    tokenizer = CharacterLevelTokenizer()
    
    # Print vocabulary
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Save the tokenizer
    tokenizer.save_vocabulary("./")
    
    # Test tokenization
    test_text = "Test OCR with numbers 123 and symbols @#$!"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Encode and decode
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

if __name__ == "__main__":
    main()