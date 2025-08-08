import google.generativeai as genai

# Hardcoded prompt
user_query="""what if im found out guilty of hiring an hitman to kill a person, whats the penalty"""

PROMPT = f"""You are a query optimization assistant for a Retrieval-Augmented Generation (RAG) system.
 Your task is to rephrase a given user query into a concise, highly effective query for a vector
 database containing sections of the Pakistan Penal Code. The optimized query should remove
 unnecessary conversational elements and focus on the core legal concepts, keywords,
 and section numbers relevant to the user's intent.
**Original Query:**
{user_query}
"""

# Configure and generate content in one go
genai.configure(api_key="AIzaSyAdO0p8tO9pyC4gnqnsvC6n4vwcDS8dzjM") 
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content(PROMPT)

print(response.text)