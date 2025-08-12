import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

user_query="""what is the penality for abducting someone?"""

PROMPT = f"""You are a query optimization assistant for a Retrieval-Augmented Generation (RAG) system.
Your task is to rephrase a given user query into a highly effective query for a vector
database containing sections of the Pakistan Penal Code. The optimized query should remove
unnecessary conversational elements and focus on the core legal concepts, keywords,
and section numbers relevant to the user's intent. When a query relates to a specific
topic, include the relevant chapter numbers from the list below to narrow the search.

**Pakistan Penal Code Chapters:**
- CHAPTER I: INTRODUCTION
- CHAPTER II: GENERAL EXPLANATIONS
- CHAPTER III: OF PUNISHMENTS
- CHAPTER IV: GENERAL EXCEPTIONS
- CHAPTER V: OF ABETMENT
- CHAPTER VI: OF OFFENCES AGAINST THE STATE
- CHAPTER VII: OF OFFENCES RELATING TO THE ARMY, NAVY AND AIR FORCE
- CHAPTER VIII: OF OFFENCES AGAINST THE PUBLIC TRANQUILLITY
- CHAPTER IX: OF OFFENCES BY OR RELATING TO PUBLIC SERVANTS
- CHAPTER X: OF CONTEMPTS OF THE LAWFUL AUTHORITY OF PUBLIC SERVANTS
- CHAPTER XI: OF FALSE EVIDENCE AND OFFENCES AGAINST PUBLIC JUSTICE
- CHAPTER XII: OF OFFENCES RELATING TO COIN AND GOVERNMENT STAMPS
- CHAPTER XIII: OF OFFENCES RELATING TO WEIGHTS AND MEASURES
- CHAPTER XIV: OF OFFENCES AFFECTING THE PUBLIC HEALTH, SAFETY, CONVENIENCE, DECENCY AND MORALS
- CHAPTER XV: OF OFFENCES RELATING TO RELIGION
- CHAPTER XVI: OF OFFENCES AFFECTING THE HUMAN BODY
- CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
- CHAPTER XVIII: OF OFFENCES RELATING TO DOCUMENTS AND TO TRADE OR PROPERTY MARKS
- CHAPTER XIX: OF THE CRIMINAL BREACH OF CONTRACTS OF SERVICE
- CHAPTER XX: OF OFFENCES RELATING TO MARRIAGE
- CHAPTER XXI: OF DEFAMATION
- CHAPTER XXII: OF CRIMINAL INTIMIDATION, INSULT AND ANNOYANCE
- CHAPTER XXIII: OF ATTEMPTS TO COMMIT OFFENCES

**Original Query:**
{user_query}
"""

# Configure and generate content in one go
genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content(PROMPT)

print(response.text)