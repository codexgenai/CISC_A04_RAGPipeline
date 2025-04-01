import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch


class EmbeddingPreparer:
    def __init__(self,
                 file_list,
                 input_dir,
                 output_dir,
                 embedding_model_name):
        self.file_list = file_list
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.embedding_model_name = embedding_model_name
        self.device = torch.device("cpu")  # Force CPU usage

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Load model and tokenizer with float32 precision for CPU
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(
            self.embedding_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()  # Set to evaluation mode

    def process_files(self):
        save_log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.INFO)

        for file_path in self.file_list:
            file_path = Path(self.input_dir/file_path)
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue
            try:
                self.logger.info(f"Processing: {file_path}")
                text = self._read_file(file_path)
                embedding = self._generate_embedding(text)
                self._save_embedding(file_path, embedding)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        logging.getLogger().setLevel(save_log_level)

    def _read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _generate_embedding(self, text):
        """Generate embedding for the given text."""
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return embeddings[0].tolist()  # Convert to list for JSON serialization
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise

    def _save_embedding(self, file_path, embedding):
        """Save embedding to a JSON file."""
        try:
            output_file = self.output_dir / f"{file_path.stem}_embeddings.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"embedding": embedding}, f)
            self.logger.info(f"Saved embeddings to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving embedding: {e}")
            raise


