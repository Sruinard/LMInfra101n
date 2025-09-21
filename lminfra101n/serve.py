import logging
import requests
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChatConfig:
    """Configuration for chat functionality."""
    max_new_tokens: int = 100
    do_sample: bool = False
    openai_base_url: str = "http://localhost:8000"
    model_name: str = "google/gemma-3-4b-it"
    timeout: int = 30


class ChatHandler:
    """Handles chat interactions with different backends."""
    
    def __init__(self, processor, model, config: ChatConfig):
        self.processor = processor
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _format_user_message(self, content: str) -> Dict[str, Any]:
        """Format user message consistently."""
        return {
            "role": "user",
            "content": content  # Simplified format for OpenAI compatibility
        }
    
    def _format_assistant_message(self, content: str) -> Dict[str, Any]:
        """Format assistant message consistently."""
        return {
            "role": "assistant", 
            "content": content
        }
    
    def _generate_hf_response(self, messages: List[Dict]) -> str:
        """Generate response using HuggingFace model."""
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample
                )
                generation = generation[0][input_len:]
            
            return self.processor.decode(generation, skip_special_tokens=True)
            
        except Exception as e:
            self.logger.error(f"HF generation failed: {e}")
            return "Sorry, I encountered an error generating a response."
    
    def _generate_openai_response(self, messages: List[Dict]) -> str:
        """Generate response using OpenAI-compatible API."""
        try:
            response = requests.post(
                f"{self.config.openai_base_url}/v1/chat/completions",
                json={
                    "model": self.config.model_name,
                    "messages": messages
                },
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return "Sorry, I couldn't connect to the API."
        except (KeyError, IndexError) as e:
            self.logger.error(f"Invalid API response format: {e}")
            return "Sorry, I received an invalid response from the API."
    
    def generate_response(self, messages: List[Dict], mode: str) -> str:
        """Generate response based on the specified mode."""
        if mode == "hf":
            return self._generate_hf_response(messages)
        elif mode == "openai":
            return self._generate_openai_response(messages)
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def get_user_input() -> Optional[str]:
    """Get user input with basic validation."""
    try:
        message = input("Enter a message (Ctrl+C to quit): ").strip()
        return message if message else None
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return None
    except EOFError:
        return None


def chat_loop(processor, model, args):
    """
    Interactive chat loop with improved error handling and structure.
    
    Args:
        processor: Text processor/tokenizer
        model: Language model 
        args: Arguments containing mode and other config
    """
    logger = logging.getLogger(__name__)
    config = ChatConfig()
    chat_handler = ChatHandler(processor, model, config)
    
    messages = []
    logger.info(f"Starting chat in {args.mode} mode...")
    
    try:
        while True:
            # Get user input
            user_message = get_user_input()
            if user_message is None:  # Handle Ctrl+C or empty input
                break
            
            # Log user input
            logger.info(f"User: {user_message}")
            
            # Add user message to conversation
            messages.append(chat_handler._format_user_message(user_message))
            
            # Generate and log assistant response
            try:
                assistant_response = chat_handler.generate_response(messages, args.mode)
                logger.info(f"Assistant: {assistant_response}")
                print(f"Assistant: {assistant_response}")
                
                # Add assistant message to conversation
                messages.append(chat_handler._format_assistant_message(assistant_response))
                
            except Exception as e:
                logger.error(f"Failed to generate response: {e}")
                print("Sorry, I encountered an error. Please try again.")
                # Remove the user message since we couldn't respond
                messages.pop()
    
    except Exception as e:
        logger.error(f"Unexpected error in chat loop: {e}")
        print("An unexpected error occurred. Exiting...")
    
    finally:
        logger.info("Chat session ended")


def create_chat_session(processor, model, config: ChatConfig):
    """Factory function to create a chat session."""
    return ChatHandler(processor, model, config)
