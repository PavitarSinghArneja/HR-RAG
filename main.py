import os
import sys
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage, AIMessage
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment_variables():
    """Check if all required environment variables are set."""
    required_vars = [
        "GOOGLE_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME"  # Added this for clarity
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    print("✅ All required environment variables are set!")
    return True

class VectorStoreManager:
    """Manages vector store operations for document retrieval."""
    
    def __init__(self, index_name: str = None):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "ragn8n")
        self.embeddings = None
        self.vectorstore = None
        self.pc = None
    
    def initialize(self):
        """Initialize Pinecone and embeddings."""
        try:
            # Check environment variables
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            
            if not pinecone_api_key:
                raise Exception("PINECONE_API_KEY environment variable not set")
            if not google_api_key:
                raise Exception("GOOGLE_API_KEY environment variable not set")
            
            print(f"🔑 Using Pinecone API Key: {pinecone_api_key[:8]}...")
            print(f"🔑 Using Google API Key: {google_api_key[:8]}...")
            print(f"📊 Using Pinecone Index: {self.index_name}")
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_api_key)
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.warning(f"⚠️ Index '{self.index_name}' not found. Available indexes: {existing_indexes}")
                raise Exception(f"Pinecone index '{self.index_name}' does not exist")
            
            # Initialize embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=google_api_key
            )
            
            # Initialize vector store
            self.vectorstore = PineconeVectorStore(
                index=self.pc.Index(self.index_name),
                embedding=self.embeddings
            )
            
            # Check index stats
            index_stats = self.pc.Index(self.index_name).describe_index_stats()
            total_vectors = index_stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                logger.warning("⚠️ No documents found in Pinecone index. Please run document ingestion first.")
            else:
                logger.info(f"📊 Found {total_vectors} document chunks in the index")
            
            logger.info("✅ Vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 3):
        """Search documents in the vector store."""
        if not self.vectorstore:
            raise Exception("Vector store not initialized")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"🔍 Found {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
        except Exception as e:
            logger.error(f"❌ Error searching documents: {e}")
            return []
    
    def get_index_stats(self):
        """Get statistics about the Pinecone index."""
        if not self.pc:
            return None
        
        try:
            index_stats = self.pc.Index(self.index_name).describe_index_stats()
            return index_stats
        except Exception as e:
            logger.error(f"❌ Error getting index stats: {e}")
            return None

class HRChatbot:
    """HR Chatbot with document retrieval capabilities."""
    
    def __init__(self):
        self.vectorstore_manager = VectorStoreManager()
        self.llm = None
        self.conversation_history = []
    
    def initialize(self):
        """Initialize all components of the chatbot."""
        try:
            print("🚀 Initializing HR Chatbot...")
            
            # Check environment variables first
            if not check_environment_variables():
                raise Exception("Missing required environment variables")
            
            # Initialize vector store
            if not self.vectorstore_manager.initialize():
                raise Exception("Failed to initialize vector store")
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7
            )
            
            print("✅ HR Chatbot initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing chatbot: {e}")
            return False
    
    def search_company_documents(self, query: str) -> str:
        """Search company documents and return formatted results."""
        try:
            results = self.vectorstore_manager.search_documents(query, k=3)
            
            if not results:
                return "No relevant documents found for your query. You may need to run document ingestion first."
            
            # Format results with better structure
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result.page_content
                
                # Truncate long content but keep it readable
                if len(content) > 800:
                    content = content[:800] + "..."
                
                source = result.metadata.get('source', 'Unknown')
                chunk_info = ""
                if 'chunk_id' in result.metadata:
                    chunk_info = f" (Part {result.metadata['chunk_id'] + 1})"
                
                formatted_results.append(
                    f"**📄 Document {i}** - {source}{chunk_info}\n"
                    f"{content}\n"
                )
            
            return "\n---\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"❌ Error searching documents: {e}")
            return f"Error searching documents: {str(e)}"
    
    def chat(self, message: str) -> str:
        """Process a chat message and return response."""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user", 
                "message": message, 
                "timestamp": datetime.now()
            })
            
            # Search for relevant documents
            document_context = self.search_company_documents(message)
            
            # Create context-aware prompt
            system_prompt = f"""You are a helpful HR assistant chatbot for our company. You have access to company documents to help answer employee questions about policies, procedures, benefits, and other HR-related topics.

Here are the most relevant documents for the user's question:

{document_context}

Previous conversation context:
{self._format_chat_history()}

Current user question: {message}

Instructions:
- Provide helpful, accurate responses based on the company documents above
- If the documents contain relevant information, use it to answer the question
- If the documents don't contain specific information needed, provide general HR guidance while noting that specific company policies weren't found
- Be conversational and friendly while remaining professional
- If asked about policies not covered in the documents, suggest contacting HR directly for the most current information
- Always prioritize information from the company documents over general knowledge

Please provide your response:"""

            # Get response from LLM
            response = self.llm.invoke([HumanMessage(content=system_prompt)])
            ai_response = response.content
            
            # Add AI response to history
            self.conversation_history.append({
                "role": "assistant", 
                "message": ai_response, 
                "timestamp": datetime.now()
            })
            
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ Error in chat: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}. Please try again or contact your HR department for assistance."
    
    def _format_chat_history(self) -> str:
        """Format conversation history for the prompt."""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted = []
        # Get last 6 messages to keep context manageable
        recent_history = self.conversation_history[-6:]
        
        for msg in recent_history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            # Truncate very long messages
            message_text = msg['message']
            if len(message_text) > 200:
                message_text = message_text[:200] + "..."
            formatted.append(f"{role}: {message_text}")
        
        return "\n".join(formatted)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("🗑️ Conversation history cleared")
    
    def get_system_status(self):
        """Get system status information."""
        status = {
            "vector_store_initialized": self.vectorstore_manager.vectorstore is not None,
            "llm_initialized": self.llm is not None,
            "conversation_history_length": len(self.conversation_history)
        }
        
        # Get index stats if available
        index_stats = self.vectorstore_manager.get_index_stats()
        if index_stats:
            status["total_documents"] = index_stats.get('total_vector_count', 0)
            status["index_name"] = self.vectorstore_manager.index_name
        
        return status

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="HR Assistant Chatbot",
        page_icon="👥",
        layout="wide"
    )
    
    st.title("🏢 HR Assistant Chatbot")
    st.markdown("Ask me anything about company policies, procedures, or HR-related questions!")
    
    # Check environment variables first
    if not check_environment_variables():
        st.error("❌ Missing required environment variables. Please check your .env file.")
        st.markdown("""
        **Required environment variables:**
        - `GOOGLE_API_KEY` - Your Google AI API key
        - `PINECONE_API_KEY` - Your Pinecone API key  
        - `PINECONE_INDEX_NAME` - Name of your Pinecone index (default: ragn8n)
        
        Create a `.env` file in your project directory with these variables.
        """)
        st.stop()
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = HRChatbot()
            if not st.session_state.chatbot.initialize():
                st.error("❌ Failed to initialize chatbot. Please check your configuration and try again.")
                st.stop()
        
        st.success("✅ Chatbot initialized successfully!")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("Chat Controls")
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chatbot.clear_history()
            st.success("Chat history cleared!")
        
        st.markdown("---")
        
        # System status
        st.header("System Status")
        if st.button("Check Status"):
            status = st.session_state.chatbot.get_system_status()
            
            st.write("**Vector Store:**", "✅" if status["vector_store_initialized"] else "❌")
            st.write("**LLM:**", "✅" if status["llm_initialized"] else "❌")
            st.write("**Chat History:**", f"{status['conversation_history_length']} messages")
            
            if "total_documents" in status:
                st.write("**Documents:**", f"{status['total_documents']} chunks")
                st.write("**Index:**", status.get('index_name', 'Unknown'))
        
        st.markdown("---")
        
        # About section
        st.header("About")
        st.markdown("""
        This HR chatbot can help you with:
        • Company policies and procedures
        • Benefits information  
        • HR-related questions
        • Document searches
        
        **Need to add documents?**
        Run the document ingestion script:
        ```bash
        python document_ingestion.py
        ```
        """)
        
        st.markdown("---")
        
        # How it works
        st.header("How it works")
        st.markdown("""
        1. **Ask your question** - Type any HR-related question
        2. **Document search** - Bot searches company documents  
        3. **AI response** - Provides answer with context from your documents
        4. **Follow up** - Continue the conversation naturally
        """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about HR policies, benefits, or any work-related questions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking..."):
                response = st.session_state.chatbot.chat(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def run_cli():
    """Run the chatbot in CLI mode for testing."""
    print("🏢 HR Assistant Chatbot - CLI Mode")
    print("=" * 50)
    
    chatbot = HRChatbot()
    if not chatbot.initialize():
        print("❌ Failed to initialize chatbot. Exiting.")
        return
    
    print("\n✅ Chatbot initialized successfully!")
    print("Type 'quit' to exit, 'clear' to clear history, 'status' for system status")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🗑️ Chat history cleared!")
                continue
            elif user_input.lower() == 'status':
                status = chatbot.get_system_status()
                print("\n📊 System Status:")
                print(f"   Vector Store: {'✅' if status['vector_store_initialized'] else '❌'}")
                print(f"   LLM: {'✅' if status['llm_initialized'] else '❌'}")
                print(f"   Chat History: {status['conversation_history_length']} messages")
                if "total_documents" in status:
                    print(f"   Documents: {status['total_documents']} chunks")
                    print(f"   Index: {status.get('index_name', 'Unknown')}")
                continue
            elif not user_input:
                continue
            
            print("\n🤖 HR Bot: ", end="")
            response = chatbot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        # This will raise an exception if not running in Streamlit
        st.session_state
        main()
    except:
        # Running in CLI mode
        if len(sys.argv) > 1 and sys.argv[1] == 'cli':
            run_cli()
        else:
            print("🏢 HR Assistant Chatbot")
            print("=" * 30)
            print("Usage:")
            print("  streamlit run main.py          # Web interface")
            print("  python main.py cli             # Command line interface")
            print("\nMake sure to run document ingestion first:")
            print("  python document_ingestion.py")