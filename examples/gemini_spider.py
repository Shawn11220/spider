import os
import spider
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
DB_PATH = "./spider_graph.db"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY environment variable")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def main():
    print("🤖 Initializing Spider + Gemini RAG...")
    
    # 1. Load Resources
    try:
        # Re-instantiate DB from disk
        db = spider.SpiderDB(DB_PATH, None, None, None)
        encoder = SentenceTransformer(EMBEDDING_MODEL)
        model = setup_gemini()
    except Exception as e:
        print(f"Error loading resources: {e}")
        print("Did you run setup_spider.py first?")
        return

    print("✅ System Ready. Ask about Rust, Space, or Biology. (Type 'quit' to exit)")

    # 2. Chat Loop
    chat = model.start_chat(history=[])
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['quit', 'exit']:
            break

        # --- A. Retrieval (The "Spider" Part) ---
        query_vec = encoder.encode(user_query).tolist()
        
        # use hybrid_search from src/db.rs
        # k=3: get top 3 most relevant nodes
        # ef_search=50: higher accuracy parameter for HNSW
        search_results = db.hybrid_search(query_vec, k=5, ef_search=100)
        
        context_texts = []
        print(f"   (🕸️  Spider found {len(search_results)} relevant memories)")
        
        for node_id, score in search_results:
            content = db.get_node(node_id)
            context_texts.append(content)
            # DEBUG:Print what the DB found to verify relevance
            print(f"   - [Score: {score:.2f}] {content[:50]}...")

        context_block = "\n".join(context_texts)

        # --- B. Generation (The "Gemini" Part) ---
        prompt = f"""
        You are an expert assistant augmented with a specific Knowledge Graph.
        
        INSTRUCTIONS:
        1. Answer the user's question explicitly.
        2. Use ONLY the provided Context Information below.
        3. Do not simply list the facts. Synthesize them into a complete paragraph.
        4. If the provided context mentions "Rust" or "Cooking", try to explain *why* it matters.
        5. If the answer is not in the context, state "I do not have that information."

        CONTEXT INFORMATION:
        {context_block}

        USER QUESTION:
        {user_query}
        
        ANSWER:
        """

        try:
            response = chat.send_message(prompt)
            print(f"Gemini: {response.text}")
        except Exception as e:
            print(f"Gemini Error: {e}")

if __name__ == "__main__":
    main()