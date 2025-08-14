import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

def test_google_ai():
    """Test Google AI API connection"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "❌ GOOGLE_API_KEY not found in environment"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello, this is a test.")
        return "✅ Google AI API working"
    except Exception as e:
        return f"❌ Google AI API error: {str(e)}"

def test_pinecone():
    """Test Pinecone connection"""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            return "❌ PINECONE_API_KEY not found in environment"
        
        pc = Pinecone(api_key=api_key)
        indexes = [index.name for index in pc.list_indexes()]
        return f"✅ Pinecone connected. Available indexes: {indexes}"
    except Exception as e:
        return f"❌ Pinecone error: {str(e)}"

def test_google_drive():
    """Test Google Drive API connection"""
    try:
        service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        
        if not service_account_file or not os.path.exists(service_account_file):
            return "❌ Google Service Account file not found"
        
        creds = Credentials.from_service_account_file(
            service_account_file,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        
        # Test by listing files
        results = service.files().list(pageSize=1).execute()
        return "✅ Google Drive API working"
    except Exception as e:
        return f"❌ Google Drive API error: {str(e)}"

def main():
    print("🧪 Testing HR Chatbot Setup...")
    print("=" * 50)
    
    # Test each component
    print("1. Testing Google AI API...")
    print(f"   {test_google_ai()}")
    
    print("\n2. Testing Pinecone...")
    print(f"   {test_pinecone()}")
    
    print("\n3. Testing Google Drive API...")
    print(f"   {test_google_drive()}")
    
    print("\n" + "=" * 50)
    
    # Check all required environment variables
    required_vars = [
        "GOOGLE_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT", 
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "GOOGLE_DRIVE_FOLDER_ID"
    ]
    
    print("4. Checking Environment Variables...")
    all_present = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Don't print sensitive values
            if "KEY" in var:
                print(f"   ✅ {var}: ***hidden***")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: Not set")
            all_present = False
    
    print("\n" + "=" * 50)
    if all_present:
        print("🎉 Setup looks good! You can now run: python main.py")
    else:
        print("⚠️  Please fix the missing environment variables before running main.py")

if __name__ == "__main__":
    main()