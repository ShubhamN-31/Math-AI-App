import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import matplotlib.pyplot as plt
import numpy as np
from sympy import sympify, symbols, lambdify

st.set_page_config(page_title="Math AI Pro", page_icon="🔢", layout="wide")

# Custom CSS for a better Logo/Header look
st.markdown(f"""
    <style>
    .main-header {{
        font-size: 50px;
        font-weight: bold;
        color: #F63366;
        margin-bottom: 0px;
    }}
    .sub-header {{
        font-size: 18px;
        color: #FAFAFA;
        margin-bottom: 30px;
    }}
    </style>
    <div>
        <span class="main-header">🔢 Math AI</span>
        <span class="sub-header"> Pro Assistant</span>
    </div>
    """, unsafe_allow_html=True)

model = OllamaLLM(model="gemma2:2b")

# Sidebar
st.sidebar.markdown("### 🛠️ Toolbox")
tool = st.sidebar.selectbox("Select Tool", ["Chat & Solve", "Function Grapher"])

if tool == "Chat & Solve":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Enter your math problem...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        template = ChatPromptTemplate.from_messages([
            ("system", "You are a professional math tutor. Always wrap mathematical formulas in dollar signs like $x^2$ for proper rendering."),
            ("human", "{question}")
        ])
        llm_chain = template | model
        
        with st.chat_message("assistant"):
            with st.spinner("Solving..."):
                response = llm_chain.invoke({"question": user_query})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

elif tool == "Function Grapher":
    st.write("### 📈 Plot a Function")
    expr_input = st.text_input("Enter an expression (e.g., x**2 + 2*x + 1):", "x**2")
    
    if expr_input:
        try:
            x = symbols('x')
            expr = sympify(expr_input)
            f = lambdify(x, expr, "numpy")
            
            x_vals = np.linspace(-10, 10, 400)
            y_vals = f(x_vals)
            
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('#0E1117') # Match theme
            ax.set_facecolor('#0E1117')
            
            ax.plot(x_vals, y_vals, label=f"y = {expr_input}", color="#F63366", linewidth=2)
            ax.axhline(0, color='white', lw=1)
            ax.axvline(0, color='white', lw=1)
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")