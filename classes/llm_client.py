import os
import logging
from typing import Dict, Any, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
from dotenv import load_dotenv

ModelType = Literal["causal", "seq2seq"]

class LLMClient:
    """Client for interacting with Hugging Face models."""
    
    def __init__(self, llm_api_url: str, llm_model_name: str, model_type: ModelType = "causal", prompt_template: str = None):
        """Initialize the LLM client.
        
        Args:
            llm_api_url: The URL of the Hugging Face API
            llm_model_name: Name of the model to use
            model_type: Type of model - "causal" for OPT or "seq2seq" for FLAN-T5
            prompt_template: Template for formatting prompts
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = llm_model_name
        self.model_type = model_type
        self.prompt_template = prompt_template
        self.device = torch.device("cpu")  # Force CPU usage
        
        # Load environment variables
        load_dotenv()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Initialize model based on model name
        if "opt" in self.model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
        elif "flan-t5" in self.model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            # FLAN-T5 specific settings
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            # Try to auto-detect the model type
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            except Exception as e:
                self.logger.warning(f"Failed to load as causal model, trying seq2seq: {str(e)}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
        
        self.model.eval()  # Set to evaluation mode
        self.logger.info(f"Initialized LLM client with model: {self.model_name} (type: {model_type})")
    
    def query(self, query: str) -> str:
        """Send a query to the model and get a response.
        
        Args:
            query: The input query string
            
        Returns:
            str: The model's response
        """
        try:
            # Prepare input based on model type
            if "flan-t5" in self.model_name.lower():
                # FLAN-T5 expects a task prefix
                query = f"Answer the following question: {query}"
            
            # Tokenize input with padding and attention mask
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response for FLAN-T5
            if "flan-t5" in self.model_name.lower():
                response = response.replace("Answer the following question:", "").strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in LLM query: {str(e)}")
            raise

