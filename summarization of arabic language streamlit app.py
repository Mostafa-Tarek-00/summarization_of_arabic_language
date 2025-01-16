import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from arabert.preprocess import ArabertPreprocessor

def main():
    st.title("Arabic Text Summarizer")

    model_name = "abdalrahmanshahrour/arabartsummarization"
    preprocessor = ArabertPreprocessor(model_name="")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    text_summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    user_input = st.text_area("Enter the paragraph you want to summarize:", height=200)
    
    if st.button("Summarize"):
        if user_input:
            text = preprocessor.preprocess(user_input)

            # Generate the summary
            result = text_summarizer(
                text,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=3,
                repetition_penalty=3.0,
                max_length=200,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )[0]["generated_text"]

            st.subheader("Input Paragraph:")
            st.write(user_input)

            st.subheader("Summary:")
            st.write(result)

if __name__ == "__main__":
    main()
