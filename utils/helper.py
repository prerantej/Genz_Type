import google.generativeai as genai
import streamlit as st
import re

def generate(prompt, top_k=1):
    # Configure Gemini
    genai.configure(api_key=st.secrets["GKEY"])

    # Use the latest stable model
    model = genai.GenerativeModel("gemini-2.5-flash")

    query = f"Suggest {top_k} possible next word (not sentence) in Gen-Z slang style for: '{prompt}'"

    response = model.generate_content(query)
    text = response.text.strip()

    # --- CLEANING STEP ---
    # Remove markdown bold (**word**) or italics (*word*)
    text = re.sub(r"\*+", "", text)

    # Remove unnecessary punctuation around single words like quotes or colons
    text = text.strip(" '\"`-:;")
    text = text.replace("(e.g.", "")
    text = text.replace("e.g.", "")
    text = text.lstrip("•-*\"'().,[]{} ").strip()

    # Keep only the first word/phrase if Gemini gives a list or sentence
    if "," in text:
        text = text.split(",")[0].strip()
    elif "\n" in text:
        text = text.split("\n")[0].strip()

    return text
