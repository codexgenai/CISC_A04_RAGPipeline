# turn off chromadb sending stats back
export ANONYMIZED_TELEMETRY=False
# # Step 01: Ingest documents
python main.py step01_ingest --input_filename all --model_type seq2seq

# Step 02: Generate embeddings
python main.py step02_generate_embeddings --input_filename all --model_type seq2seq

# Step 03: Store vectors
python main.py step03_store_vectors --input_filename all --model_type seq2seq

# Step 04: Test retrieval
python main.py step04_retrieve_chunks --query_args "What is massachusetts's right to know law?" --model_type seq2seq

# Step 05: Generate response
python main.py step05_generate_response --query_args "What is massachusetts's right to know law?" --use_rag --model_type seq2seq