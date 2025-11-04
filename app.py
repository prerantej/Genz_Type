import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.helper import generate
import re

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="GenZ-Type ", page_icon="🔥", layout="centered")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
            /* Hide top-right GitHub button */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    /* Optional: also hide deploy/share menu if you want a cleaner look */
    [data-testid="stDecoration"] {
        display: none !important;
    }
html, body, [data-testid="stAppViewContainer"], .stApp {
    height: 100%;
    width: 100%;
    /* Modern black & white diagonal gradient */
      background: radial-gradient(
        circle at 10% 40%,
        rgba(255,255,255,0.10) 0%,
        rgba(255,255,255,0.05) 20%,
        rgba(0,0,0,0.95) 85%,
        rgba(0,0,0,1) 100%
    ),
    radial-gradient(
        circle at 85% 95%,
        rgba(255,255,255,0.05) 23%,
        rgba(0,0,0,0.95) 70%
    );
    background-blend-mode: overlay;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-size: cover;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    background: -webkit-linear-gradient(45deg, #ff00cc, #3333ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-weight: 800;
}
.stTextInput>div>div>input {
    background-color: #1e1e1e;
    color: #fff;
    border: 1px solid #ff00cc;
    border-radius: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #ff00cc, #3333ff);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 15px #ff00cc;
}
.suggestion {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #ff00cc;
    border-radius: 8px;
    padding: 8px 15px;
    margin: 4px;
    cursor: pointer;
    font-weight: 600;
}
.suggestion:hover {
    background: #ff00cc;
    color: white;
}
header[data-testid="stHeader"] {
    background: black !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title(" GenZ-Type: Next Word Predictor ")
st.caption("A slang-aware text generator trained on real Gen-Z conversations ")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model_path = "PreranTej/genz-slang-causal-model-final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model

with st.spinner("Loading GenZ model... please wait ⏳"):
    tokenizer, model = load_model()

# -------------------- INPUT --------------------
st.subheader("Type your text 👇")
user_input = st.text_input("Your text:", "nah cause this vibe straight")

# -------------------- GENERATE SUGGESTIONS --------------------
# -------------------- GENERATE SUGGESTIONS --------------------
if st.button("✨ Generate Suggestions"):
    if user_input.strip():
        suggestions = []

        # ---- Local model: get top-k next words ----
        try:
            inputs = tokenizer(user_input, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]

                # get top-k indices instead of just one
                top_k = 3   # 👈 you can change this number for more suggestions
                top_tokens = torch.topk(next_token_logits, top_k).indices.tolist()

                for tok in top_tokens:
                    word = tokenizer.decode([tok]).strip()
                    # filter punctuation-only tokens
                    if word and not re.match(r"^[\W_]+$", word):
                        suggestions.append(word)
        except Exception as e:
            st.error(f"Model error: {e}")

        # ---- Gemini generation ----
        try:
            gsuggestion = generate(user_input, top_k=1)
            if gsuggestion and not re.match(r"^[\W_]+$", gsuggestion):
                suggestions.append(gsuggestion)
        except Exception as e:
            st.error(f"Gemini error: {e}")

        # ---- Display results ----
        if suggestions:
            st.markdown("<h4> Next-word suggestions:</h4>", unsafe_allow_html=True)
            cols = st.columns(len(suggestions))
            for i, word in enumerate(suggestions):
                with cols[i]:
                    st.markdown(f"<div class='suggestion'>{word}</div>", unsafe_allow_html=True)
        else:
            st.warning("No valid suggestions generated ")

    else:
        st.warning("Please type some text first ")


st.markdown("---")
st.caption("💡 Tip: Try prompts like *'the party was'* or *'this outfit kinda '* for Gen-Z slang vibes.")
