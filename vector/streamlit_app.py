import streamlit as st
from function_calling_crew import get_crew_response
import time
import sys
from io import StringIO
import contextlib

# Set page config
st.set_page_config(
    page_title="Meeting Assistant",
    page_icon="🤖",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .output-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 AI Meeting Assistant")
st.markdown("""
This assistant can help you:
- Search through meeting recordings
- Analyze meeting content
- Perform calculations related to meetings
""")

# Create tabs
tab1, tab2 = st.tabs(["Chat Interface", "About"])

# Add this function to show processing steps
def show_processing_steps():
    steps = [
        ("🔍 Research Phase", [
            "Analyzing query intent...",
            "Searching through meeting recordings...",
            "Collecting relevant information..."
        ]),
        ("🧠 Analysis Phase", [
            "Processing search results...",
            "Analyzing meeting content...",
            "Extracting key insights..."
        ]),
        ("📝 Synthesis Phase", [
            "Combining information...",
            "Formatting response...",
            "Preparing final answer..."
        ])
    ]
    
    progress_placeholder = st.empty()
    steps_placeholder = st.empty()
    
    for i, (phase, substeps) in enumerate(steps):
        progress = (i + 1) / len(steps)
        progress_placeholder.progress(progress)
        
        with steps_placeholder.container():
            st.markdown(f"**Current Phase: {phase}**")
            for substep in substeps:
                time.sleep(0.5)  # Add small delay for visual effect
                st.markdown(f"- {substep}")
        
        time.sleep(1)  # Pause between phases
    
    progress_placeholder.empty()
    steps_placeholder.empty()

class ConsoleOutput:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.output = []
        self.current_section = []
        self.current_section_type = None

    def write(self, text):
        # Skip empty lines and unwanted messages
        if not text.strip() or "Overriding of current TracerProvider" in text:
            return
            
        # Clean up the text
        text = text.replace('[00m', '').replace('[92m', '')
        
        # Check if this is a new section
        section_markers = {
            'Agent:': '🤖 AGENT',
            'Task:': '📋 TASK',
            'Tool Input:': '📥 INPUT',
            'Tool Output:': '📤 OUTPUT',
            'Thought:': '💭 THOUGHT'
        }
        
        for marker, header in section_markers.items():
            if marker in text:
                # If we have a previous section, add it to output
                if self.current_section:
                    self.output.append('\n'.join(self.current_section))
                    self.output.append('\n' + '-'*50 + '\n')  # Add separator
                
                # Start new section
                self.current_section = [f"### {header} ###"]
                self.current_section_type = marker
                text = text.replace(marker, '').strip()
                break
        
        # Format JSON-like content
        if text.strip().startswith('{') or text.strip().startswith('['):
            try:
                import json
                parsed = json.loads(text)
                text = json.dumps(parsed, indent=2)
            except:
                pass
        
        # Wrap long lines (except for JSON content)
        if not (text.strip().startswith('{') or text.strip().startswith('[')):
            import textwrap
            wrapped_lines = []
            for line in text.split('\n'):
                # Wrap at 80 characters, preserving indentation
                wrapped = textwrap.fill(line, width=80, subsequent_indent='    ')
                wrapped_lines.append(wrapped)
            text = '\n'.join(wrapped_lines)
        
        # Add text to current section
        self.current_section.append(text)
        
        # Format and display the complete output
        full_output = '\n'.join([
            *self.output,
            '\n'.join(self.current_section)
        ])
        
        # Create a container with custom CSS for better text wrapping
        self.placeholder.markdown(
            f"""<div style='font-family: monospace; white-space: pre-wrap; 
            word-wrap: break-word; overflow-wrap: break-word; 
            max-width: 100%; padding: 10px; background-color: #f0f2f6; 
            border-radius: 5px;'>
            {full_output.replace('{', '{{').replace('}', '}}')}</div>""", 
            unsafe_allow_html=True
        )

    def flush(self):
        pass

with tab1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about the meetings?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response with loading indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            progress_bar = st.progress(0)
            console_placeholder = st.empty()
            
            try:
                # Create custom console output handler
                console_output = ConsoleOutput(console_placeholder)
                
                # Redirect stdout and stderr to our custom handler
                with contextlib.redirect_stdout(console_output), contextlib.redirect_stderr(console_output):
                    progress_bar.progress(0.3)
                    full_response = get_crew_response(prompt)
                    progress_bar.progress(1.0)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Show response
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.error(error_message)
                console_placeholder.code(f"Error occurred:\n{str(e)}")
            
            finally:
                # Clean up
                progress_bar.empty()

with tab2:
    st.header("About this Assistant")
    st.markdown("""
    ### Capabilities
    This AI assistant uses multiple tools to help you understand your meeting recordings:

    1. **Search Tool**: 
       - Searches through meeting recordings using semantic similarity
       - Finds relevant discussions and topics

    2. **Analysis Tool**: 
       - Analyzes meeting content in detail
       - Extracts key points and decisions
       - Evaluates sentiment and importance

    3. **Calculator Tool**: 
       - Performs calculations related to meetings
       - Can analyze durations, frequencies, etc.

    ### How to Use
    Simply type your question in the chat interface. The assistant will:
    1. Understand your query
    2. Choose the appropriate tools
    3. Process the information
    4. Provide a clear, structured response

    ### Examples
    Try asking questions like:
    - "What was discussed in recent API meetings?"
    - "Analyze the key decisions from the last sales meeting"
    - "What's the total duration of all product meetings?"
    """)

    st.markdown("---")
    st.markdown("### Technical Details")
    st.code("""
    Technologies used:
    - CrewAI for agent orchestration
    - Qdrant for vector search
    - OpenAI for embeddings
    - Anthropic's Claude for analysis
    - Sentence Transformers for encoding
    """)

# Add a sidebar with additional controls if needed
with st.sidebar:
    st.header("Settings")
    
    # Add any configuration options
    st.subheader("Search Settings")
    search_limit = st.slider("Number of results", 1, 10, 5)
    
    st.subheader("Analysis Settings")
    analysis_depth = st.select_slider(
        "Analysis Depth",
        options=["Basic", "Standard", "Detailed"],
        value="Standard"
    )
    
    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Add version info
    st.markdown("---")
    st.markdown("v1.0.0") 