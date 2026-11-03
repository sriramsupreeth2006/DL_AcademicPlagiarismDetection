import streamlit as st

from predict import predict


st.set_page_config(
    page_title="Academic Plagiarism Detection",
    page_icon="📚",
    layout="centered"
)

st.title("Academic Plagiarism Detection")

st.write(
    "Enter two sentences to check whether they have similar meanings."
)

sentence1 = st.text_area(
    "Enter First Sentence"
)

sentence2 = st.text_area(
    "Enter Second Sentence"
)


if st.button("Check Similarity"):

    if sentence1.strip() == "" or sentence2.strip() == "":

        st.warning("Please enter both sentences.")

    else:

        with st.spinner("Analyzing sentences..."):

            score, result = predict(sentence1, sentence2)

        st.subheader("Prediction Result")

        st.metric(
            label="Paraphrase Probability",
            value=f"{score:.2f}%"
        )

        if result == "Paraphrase":

            st.success("The sentences are classified as Paraphrases.")

        else:

            st.error("The sentences are classified as Not Paraphrases.")