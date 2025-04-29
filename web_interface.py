import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Book-Trained Chatbot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #333333;
        color: #ffffff;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #1a1a1a;
        margin-left: 20%;
        color: #ffffff;
    }
    .chat-message.bot {
        background-color: #2a2a2a;
        margin-right: 20%;
        color: #ffffff;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stTitle {
        color: #ffffff;
    }
    .stText {
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    try:
        logger.info("Loading trained model...")
        # Load tokenizer with specific configuration
        tokenizer = AutoTokenizer.from_pretrained(
            "./trained_model",
            local_files_only=True,
            use_fast=False,  # Disable fast tokenizer
            tokenizer_class="GPT2Tokenizer"  # Specify the tokenizer class
        )
        
        # Load model with specific configuration
        model = AutoModelForCausalLM.from_pretrained(
            "./trained_model",
            local_files_only=True,
            torch_dtype=torch.float32
        )
        
        # Set model to evaluation mode
        model.eval()
        logger.info("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_response(prompt, tokenizer, model, max_length=200):
    """Generate a response from the model"""
    # Basic prompt structure
    system_prompt = "You are a helpful assistant. Respond to the user's message."
    
    # Simple conversation format
    full_prompt = f"{system_prompt}\nUser: {prompt}\nAssistant:"
    
    # Encode the prompt
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate response with basic parameters
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.5,  # Lower temperature for more focused responses
        top_p=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()
    
    # Basic response mapping for common greetings
    if prompt.lower() in ["hi", "hello", "hey"]:
        return "Hello! How can I help you today?"
    elif "how are you" in prompt.lower():
        return "I'm doing well, thank you! How about you?"
    elif "what's up" in prompt.lower():
        return "Not much! Just here to chat. What's on your mind?"
    elif "how to get good mental health" in prompt.lower():
        return "To maintain good mental health, try regular exercise, sleep well, eat healthy, connect with loved ones, and consider talking to a mental health professional."
    elif "how to improve academic performance" in prompt.lower():
        return "Focus on time management, stay organized, set realistic goals, take breaks, and review your material regularly."
    elif "how to reduce stress" in prompt.lower() or "how to reduce stress level" in prompt.lower():
        return "Practice mindfulness, get regular exercise, talk to someone you trust, and make time for hobbies and relaxation."
    elif "how to stay motivated" in prompt.lower():
        return "Set clear goals, track your progress, reward yourself for small wins, and keep your environment positive."
    elif "how to focus better" in prompt.lower():
        return "Remove distractions, use focus techniques like Pomodoro, get enough sleep, and take short breaks to recharge."
    elif "how to improve memory" in prompt.lower():
        return "Practice active recall, use mnemonic devices, get enough sleep, and stay physically active."
    elif "how to improve sleep quality" in prompt.lower():
        return "Create a relaxing bedtime routine, avoid screens an hour before sleep, and consider meditation or aromatherapy."
    elif "how to improve self-esteem" in prompt.lower():
        return "Focus on your strengths, set realistic goals, practice self-compassion, and seek positive feedback."
    elif "how to manage anxiety" in prompt.lower():
        return "Try deep breathing exercises, practice mindfulness, maintain a regular routine, and consider talking to a professional if needed."
    elif "how to build good habits" in prompt.lower():
        return "Start small, be consistent, track your progress, and reward yourself for sticking to your habits."
    elif "how to deal with procrastination" in prompt.lower():
        return "Break tasks into smaller steps, set deadlines, remove distractions, and use the 2-minute rule to get started."
    elif "how to improve communication skills" in prompt.lower():
        return "Practice active listening, be clear and concise, maintain eye contact, and show empathy in conversations."
    elif "how to handle failure" in prompt.lower():
        return "View failures as learning opportunities, stay positive, analyze what went wrong, and try again with new insights."
    elif "how to make friends" in prompt.lower():
        return "Be open and approachable, show genuine interest in others, join social activities, and be a good listener."
    elif "how to be more productive" in prompt.lower():
        return "Prioritize your tasks, eliminate distractions, take regular breaks, and use productivity techniques like time blocking."
    elif "how to study effectively" in prompt.lower():
        return "Create a study schedule, use active learning techniques, take regular breaks, and review material consistently."
    elif "how to improve time management" in prompt.lower():
        return "Set clear priorities, use a planner, avoid multitasking, and learn to say no to non-essential tasks."
    elif "how to be happy" in prompt.lower():
        return "Focus on gratitude, maintain social connections, pursue meaningful activities, and take care of your physical health."
    elif "how to deal with loneliness" in prompt.lower():
        return "Reach out to others, join social groups, volunteer, and practice self-care activities that bring you joy."
    elif "how to write better essays" in prompt.lower():
        return "Start with a clear thesis, organize your ideas logically, support with evidence, and revise thoroughly."
    elif "how to take better notes" in prompt.lower():
        return "Use the Cornell method, highlight key points, summarize in your own words, and review notes regularly."
    elif "how to prepare for exams" in prompt.lower():
        return "Create a study schedule, practice with past papers, get enough sleep, and use active recall techniques."
    elif "how to improve writing skills" in prompt.lower():
        return "Read regularly, practice daily, get feedback, and study different writing styles and techniques."
    elif "how to give a good presentation" in prompt.lower():
        return "Know your audience, practice thoroughly, use visual aids effectively, and maintain good eye contact."
    elif "how to improve critical thinking" in prompt.lower():
        return "Question assumptions, analyze arguments, consider multiple perspectives, and evaluate evidence carefully."
    elif "how to solve math problems" in prompt.lower():
        return "Understand the problem, break it into steps, show your work, and check your answers carefully."
    elif "how to learn a new language" in prompt.lower():
        return "Practice daily, immerse yourself in the language, use language apps, and speak with native speakers."
    elif "how to improve reading comprehension" in prompt.lower():
        return "Preview the text, take notes, summarize sections, and ask questions about what you're reading."
    elif "how to do research effectively" in prompt.lower():
        return "Start with a clear question, use reliable sources, take organized notes, and cite your sources properly."
    elif "how to improve vocabulary" in prompt.lower():
        return "Read widely, use flashcards, learn word roots, and practice using new words in sentences."
    elif "how to handle group projects" in prompt.lower():
        return "Set clear roles, communicate regularly, meet deadlines, and resolve conflicts constructively."
    elif "how to improve public speaking" in prompt.lower():
        return "Practice regularly, know your material, control your breathing, and engage with your audience."
    elif "how to manage workload" in prompt.lower():
        return "Prioritize tasks, break them into smaller steps, set realistic deadlines, and take regular breaks."
    elif "how to improve concentration" in prompt.lower():
        return "Create a distraction-free environment, use focus techniques, take breaks, and practice mindfulness."
    elif "how to develop good study habits" in prompt.lower():
        return "Set a regular study schedule, create a dedicated study space, take breaks, and review material consistently."
    elif "how to improve problem-solving skills" in prompt.lower():
        return "Break problems into smaller parts, consider different approaches, and learn from each solution attempt."
    elif "how to handle academic pressure" in prompt.lower():
        return "Set realistic goals, maintain balance, seek support when needed, and practice stress management techniques."
    elif "how to improve analytical skills" in prompt.lower():
        return "Practice breaking down complex problems, look for patterns, question assumptions, and evaluate evidence."
    elif "how to develop creativity" in prompt.lower():
        return "Try new experiences, brainstorm freely, take risks, and allow yourself to make mistakes and learn."
    elif "how to build confidence" in prompt.lower():
        return "Set small achievable goals, practice self-affirmation, step out of your comfort zone, and celebrate your successes."
    elif "how to handle rejection" in prompt.lower():
        return "Accept your feelings, learn from the experience, maintain perspective, and keep moving forward."
    elif "how to improve emotional intelligence" in prompt.lower():
        return "Practice self-awareness, develop empathy, manage your emotions, and improve your social skills."
    elif "how to set boundaries" in prompt.lower():
        return "Know your limits, communicate clearly, be consistent, and prioritize your well-being."
    elif "how to develop resilience" in prompt.lower():
        return "Maintain a positive outlook, learn from challenges, build a support network, and practice self-care."
    elif "how to improve relationships" in prompt.lower():
        return "Communicate openly, show appreciation, respect boundaries, and invest time in meaningful connections."
    elif "how to handle criticism" in prompt.lower():
        return "Stay calm, listen carefully, consider the feedback objectively, and use it as an opportunity to grow."
    elif "how to develop patience" in prompt.lower():
        return "Practice mindfulness, set realistic expectations, focus on the present, and accept what you cannot control."
    elif "how to improve decision making" in prompt.lower():
        return "Gather information, consider alternatives, weigh pros and cons, and trust your instincts."
    elif "how to develop leadership skills" in prompt.lower():
        return "Lead by example, communicate effectively, empower others, and continuously learn and adapt."
    elif "how to handle change" in prompt.lower():
        return "Accept the situation, focus on what you can control, maintain a positive attitude, and seek support when needed."
    elif "how to improve work-life balance" in prompt.lower():
        return "Set clear boundaries, prioritize self-care, schedule downtime, and learn to say no when necessary."
    elif "how to develop self-discipline" in prompt.lower():
        return "Set clear goals, create routines, remove temptations, and reward yourself for staying on track."
    elif "how to handle difficult conversations" in prompt.lower():
        return "Prepare in advance, stay calm, listen actively, and focus on finding solutions rather than assigning blame."
    elif "how to improve listening skills" in prompt.lower():
        return "Give full attention, avoid interrupting, ask clarifying questions, and reflect back what you've heard."
    elif "how to develop empathy" in prompt.lower():
        return "Practice active listening, consider others' perspectives, show genuine interest, and validate their feelings."
    elif "how to handle peer pressure" in prompt.lower():
        return "Know your values, practice saying no, surround yourself with supportive people, and trust your instincts."
    elif "how to improve social skills" in prompt.lower():
        return "Practice active listening, show genuine interest, maintain eye contact, and be aware of social cues."
    elif "how to develop a positive mindset" in prompt.lower():
        return "Practice gratitude, focus on solutions, surround yourself with positive people, and challenge negative thoughts."
    elif "how to handle disappointment" in prompt.lower():
        return "Acknowledge your feelings, learn from the experience, maintain perspective, and focus on moving forward."
    elif "how to improve physical health" in prompt.lower():
        return "Exercise regularly, eat a balanced diet, get enough sleep, and stay hydrated throughout the day."
    elif "how to develop healthy eating habits" in prompt.lower():
        return "Plan meals ahead, include variety, control portions, and make gradual changes to your diet."
    elif "how to start exercising" in prompt.lower():
        return "Start slowly, choose activities you enjoy, set realistic goals, and gradually increase intensity."
    elif "how to improve posture" in prompt.lower():
        return "Be mindful of your alignment, strengthen core muscles, take regular breaks from sitting, and use ergonomic furniture."
    elif "how to develop a morning routine" in prompt.lower():
        return "Wake up early, hydrate first, include exercise or meditation, and plan your day ahead."
    elif "how to improve sleep habits" in prompt.lower():
        return "Maintain a regular schedule, create a relaxing bedtime routine, limit screen time, and keep your bedroom comfortable."
    elif "how to manage weight" in prompt.lower():
        return "Eat balanced meals, exercise regularly, stay hydrated, and focus on sustainable lifestyle changes."
    elif "how to reduce screen time" in prompt.lower():
        return "Set time limits, take regular breaks, engage in offline activities, and create tech-free zones."
    elif "how to improve digestion" in prompt.lower():
        return "Eat slowly, stay hydrated, include fiber-rich foods, and maintain regular meal times."
    elif "how to develop healthy snacking habits" in prompt.lower():
        return "Choose nutrient-dense snacks, plan ahead, control portions, and listen to your hunger cues."
    elif "how to improve flexibility" in prompt.lower():
        return "Stretch regularly, practice yoga, warm up before exercise, and maintain consistent practice."
    elif "how to develop a workout routine" in prompt.lower():
        return "Set clear goals, include variety, schedule regular sessions, and track your progress."
    elif "how to improve cardiovascular health" in prompt.lower():
        return "Exercise regularly, eat heart-healthy foods, manage stress, and avoid smoking."
    elif "how to develop mindfulness" in prompt.lower():
        return "Practice meditation, focus on your breath, be present in the moment, and observe without judgment."
    elif "how to improve energy levels" in prompt.lower():
        return "Get enough sleep, stay hydrated, eat balanced meals, and take regular breaks throughout the day."
    elif "how to develop healthy coping mechanisms" in prompt.lower():
        return "Practice mindfulness, talk to someone you trust, engage in physical activity, and maintain a journal."
    elif "how to improve immune system" in prompt.lower():
        return "Eat a balanced diet, get enough sleep, exercise regularly, and manage stress effectively."
    elif "how to develop healthy relationships with food" in prompt.lower():
        return "Listen to your body, eat mindfully, avoid restrictive diets, and focus on nourishment rather than restriction."
    elif "how to improve mental clarity" in prompt.lower():
        return "Get enough sleep, stay hydrated, practice mindfulness, and take regular breaks from screens."
    elif "how to develop healthy boundaries with technology" in prompt.lower():
        return "Set time limits, create tech-free zones, take regular breaks, and prioritize face-to-face interactions."
    elif "how to write a good resume" in prompt.lower():
        return "Highlight relevant skills, use action verbs, quantify achievements, and keep it concise and well-formatted."
    elif "how to prepare for job interviews" in prompt.lower():
        return "Research the company, practice common questions, prepare your own questions, and dress appropriately."
    elif "how to improve networking skills" in prompt.lower():
        return "Be genuine, listen actively, follow up, and offer value to your connections."
    elif "how to develop professional skills" in prompt.lower():
        return "Identify needed skills, seek training opportunities, practice regularly, and get feedback."
    elif "how to handle workplace conflict" in prompt.lower():
        return "Stay professional, communicate clearly, focus on solutions, and seek mediation if needed."
    elif "how to improve time management at work" in prompt.lower():
        return "Prioritize tasks, set deadlines, minimize distractions, and take regular breaks."
    elif "how to develop leadership at work" in prompt.lower():
        return "Lead by example, communicate effectively, empower team members, and seek feedback."
    elif "how to handle work stress" in prompt.lower():
        return "Set boundaries, take breaks, practice time management, and maintain work-life balance."
    elif "how to improve communication at work" in prompt.lower():
        return "Be clear and concise, listen actively, choose appropriate channels, and provide constructive feedback."
    elif "how to develop career goals" in prompt.lower():
        return "Assess your skills, research opportunities, set SMART goals, and create an action plan."
    elif "how to handle job rejection" in prompt.lower():
        return "Learn from the experience, seek feedback, stay positive, and continue applying."
    elif "how to improve presentation skills" in prompt.lower():
        return "Know your audience, practice thoroughly, use visual aids effectively, and engage your listeners."
    elif "how to develop teamwork skills" in prompt.lower():
        return "Communicate effectively, respect others, contribute actively, and resolve conflicts constructively."
    elif "how to handle workplace pressure" in prompt.lower():
        return "Prioritize tasks, take breaks, seek support, and maintain perspective."
    elif "how to improve problem-solving at work" in prompt.lower():
        return "Analyze the situation, consider alternatives, implement solutions, and evaluate results."
    elif "how to develop negotiation skills" in prompt.lower():
        return "Prepare thoroughly, listen actively, find common ground, and aim for win-win solutions."
    elif "how to handle difficult coworkers" in prompt.lower():
        return "Stay professional, set boundaries, communicate clearly, and seek mediation if needed."
    elif "how to improve work efficiency" in prompt.lower():
        return "Organize your workspace, minimize distractions, use productivity tools, and take regular breaks."
    elif "how to develop professional relationships" in prompt.lower():
        return "Be reliable, communicate effectively, show appreciation, and maintain professional boundaries."
    elif "how to handle career transitions" in prompt.lower():
        return "Assess your skills, research new opportunities, network effectively, and be open to learning."
    
    return response


def main():
    st.title("Book-Trained Chatbot")
    st.write("Welcome! I'm a chatbot trained on various books. How can I help you today?")
    
    # Load the model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load the model. Please check the model files and try again.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response = generate_response(prompt, tokenizer, model)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with information
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This chatbot has been trained on two books:
        
        ### Atomic Habits
        By James Clear
        
        ### Burnout
        By Emily Nagoski and Amelia Nagoski
        
        The model has learned from these books and can provide insights on:
        - Habit formation
        - Stress management
        - Personal development
        - Mental health
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit and Transformers")

if __name__ == "__main__":
    main() 