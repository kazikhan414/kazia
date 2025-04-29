import os
from dotenv import load_dotenv
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BookTrainer:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        logger.info(f"Initializing trainer with model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set up padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def load_books(self, book_paths):
        """Load books from various formats (txt, pdf)"""
        documents = []
        for path in tqdm(book_paths, desc="Loading books"):
            try:
                if path.endswith('.txt'):
                    loader = TextLoader(path)
                elif path.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                else:
                    logger.warning(f"Unsupported file format: {path}")
                    continue
                documents.extend(loader.load())
                logger.info(f"Successfully loaded: {path}")
            except Exception as e:
                logger.error(f"Error loading {path}: {str(e)}")
        return documents

    def preprocess_text(self, documents):
        """Split text into chunks and prepare for training"""
        logger.info("Preprocessing text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks

    def create_dataset(self, chunks):
        """Create a dataset for training"""
        logger.info("Creating dataset...")
        texts = [chunk.page_content for chunk in chunks]
        dataset = Dataset.from_dict({"text": texts})
        logger.info(f"Dataset created with {len(dataset)} examples")
        return dataset

    def tokenize_function(self, examples):
        """Tokenize the text for training"""
        outputs = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors=None  # Return as lists
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    def train(self, dataset, output_dir="./trained_model"):
        """Train the model on the book data"""
        logger.info("Starting training process...")
        try:
            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=["text"]
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=1,  # Minimal batch size
                save_steps=1000,
                save_total_limit=2,
                learning_rate=2e-5,
                weight_decay=0.01,
                logging_steps=100,
                logging_dir="./logs",
                gradient_accumulation_steps=8,  # Increased gradient accumulation
                use_cpu=True  # Force CPU usage
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )

            logger.info("Training started...")
            trainer.train()
            logger.info("Training completed!")
            
            logger.info(f"Saving model to {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

def main():
    # Initialize trainer
    trainer = BookTrainer()
    
    # List of book paths to train on
    book_paths = [
        os.path.join("books", "atomic_habits.pdf"),
        os.path.join("books", "burnout.pdf")
    ]
    
    # Check if books directory exists
    if not os.path.exists("books"):
        logger.error("Books directory not found. Please create a 'books' directory and add your book files.")
        return
    
    # Check if any books exist
    if not any(os.path.exists(path) for path in book_paths):
        logger.error("No book files found in the books directory. Please add your book files.")
        return
    
    try:
        # Load and preprocess books
        documents = trainer.load_books(book_paths)
        if not documents:
            logger.error("No documents were loaded successfully.")
            return
            
        chunks = trainer.preprocess_text(documents)
        dataset = trainer.create_dataset(chunks)
        
        # Train the model
        trainer.train(dataset)
        
        logger.info("Training completed successfully! Model saved to ./trained_model")
    except Exception as e:
        logger.error(f"An error occurred during the training process: {str(e)}")

if __name__ == "__main__":
    main() 