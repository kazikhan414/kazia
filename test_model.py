from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Load the trained model and tokenizer"""
    logger.info("Loading trained model...")
    tokenizer = AutoTokenizer.from_pretrained("./trained_model")
    model = AutoModelForCausalLM.from_pretrained("./trained_model")
    logger.info("Model loaded successfully!")
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_length=200):
    """Generate a response from the model"""
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Load the model
    tokenizer, model = load_model()
    
    print("\nWelcome to the Book-Trained Chatbot!")
    print("This bot has been trained on 'Atomic Habits' and 'Burnout: The Secret to Unlocking the Stress Cycle'")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Generate response
        try:
            response = generate_response(user_input, tokenizer, model)
            print("\nBot:", response)
            print()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            print("Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    main() 