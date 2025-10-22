import streamlit as st
from answer import retrieve, build_prompt, generate

st.title("SchoLAS")
st.text("Scholarly Learning Analytics Synthesizer")

query = st.text_area("Ask me anything realated to LA Research!")

if query:
    hits = retrieve(query)

    if not hits:
        st.warning("Sorry, I don't have relevant answer to your question. 🫤")
    else: 
        prompt = build_prompt(query, hits)
        with st.spinner("Generating response..."):
            answer = generate(prompt)
        st.subheader("SchoLA's Response: ")
        st.write(answer)

        st.subheader("Top Matches")
        for i, (_, meta, score) in enumerate(hits, 1):
            st.write(f"[{i}] {meta['source_file']} p.{meta['page_number']} — score={score:.3f}")

