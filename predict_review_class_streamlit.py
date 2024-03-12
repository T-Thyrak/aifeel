import streamlit as st

import nltk
import ssl

# im Herz, verbannt
# this has to be ran first :smh:

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("average_perceptron_tagger")


# Load the model
import dill as model_file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


from aifeel.util.preprocess import preprocess_text
from aifeel.util.feature_extraction import extract_features, feature_to_vector
from aifeel.util import gen_dataframe, read_corpus
from aifeel.model.nn import NNClassifier
from scipy.sparse import hstack

# from tensorflow.keras.models import load_model
# from tensorflow.keras.saving import pickle_utils


@st.cache_resource
def init_stuffs():
    print("Exporting basic model")
    NNClassifier.export_basic_model()
    print("Exported basic model")
    return True


negative_words, positive_words = set(read_corpus("negative-words")), set(
    read_corpus("positive-words")
)

_not_used = init_stuffs()

vectorizer_for_multi = model_file.load(
    open("export/model/TFIDFModelClassifier/vectorizer.dill", "rb")
)
cv = model_file.load(open("export/model/NNClassifier/vectorizer.dill", "rb"))


# model = model_file.load(open('model.dll', 'rb'))
def visualize_probabilities(positive_prob, negative_prob):
    labels = ["Positive", "Negative"]
    sizes = [positive_prob, negative_prob]
    colors = ["#99ff99", "#f55174"]  # Blue for Positive, Green for Negative

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.2f%%", startangle=90, colors=colors)
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.legend(
        labels, loc="upper right", bbox_to_anchor=(0.5, 0, 0.5, 1)
    )  # location of legend 1=right, 2=left, 3=upper, 4=lower

    st.pyplot(fig)


def vectorizer(review):
    result = cv.transform([review])
    return result.toarray()[0].tolist()


def count_sentiment_words(review, sentiment_words):
    words = review.split()
    return sum(1 for word in words if word in sentiment_words)


def predict_review(model, reviews):
    X_tfidf = vectorizer_for_multi.transform(reviews)

    positive_word_count = [
        count_sentiment_words(review, positive_words) for review in reviews
    ]
    negative_word_count = [
        count_sentiment_words(review, negative_words) for review in reviews
    ]

    # Combine the tf-idf features with the sentiment word count features
    X = hstack([X_tfidf, np.array([positive_word_count, negative_word_count]).T])

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    return y_pred, y_prob


def main():
    st.title("Predict Review Message Category")
    # st.subheader('Enter the review message to predict the category')
    model_selection = st.selectbox(
        "Select a model:", ["NNClassifier", "SVM", "MultiNomialNB"]
    )
    # review text with placeholder
    review = st.text_area("Enter the review message:")
    positive_prob = 0
    negative_prob = 0
    if st.button("Predict"):
        st.subheader("Predicted Result:", divider="rainbow")
        if model_selection == "NNClassifier":
            with open("export/model/NNClassifier/model.dill", "rb") as f:
                model = model_file.load(f)
            text_review = preprocess_text(review)
            text_feature = extract_features(
                text_review, positive_words, negative_words, vectorizer=vectorizer
            )
            text_feature_vector = feature_to_vector(text_feature, vectorizer=True)
            result = model.predict_proba([text_feature_vector])[0]
            positive_prob = result[1]
            negative_prob = result[0]
        elif model_selection == "SVM":
            with open("export/model/SVM/svm_model.dill", "rb") as f:
                model = model_file.load(f)

            text_review = preprocess_text(review)
            text_feature = extract_features(text_review, positive_words, negative_words)
            test_review_df = pd.DataFrame([text_feature])
            result = model.predict_proba(test_review_df)[0]
            positive_prob = result[1]
            negative_prob = result[0]

        else:
            with open(
                "export/model/TFIDFModelClassifier/multinomial_nb_model.dill", "rb"
            ) as f:
                model = model_file.load(f)

            result = predict_review(model, [review])
            print(f"result:{result[1][0]}")
            positive_prob = result[1][0][1]
            negative_prob = result[1][0][0]

        visualize_probabilities(positive_prob, negative_prob)


if __name__ == "__main__":
    st.subheader("Our Team", divider="rainbow")
    st.markdown(
        f"<div>"
        f"<h4>Team 3 - Mini Project 2 - 2024 &copy; All rights reserved❤️</h4>"
        f"<ol>"
        f"<li>VUTHY Panha</li>"
        f"<li>TENG Thaisothyrak</li>"
        f"<li>SIM Daro</li>"
        f"<li>TAING Molika</li>"
        f"<li>PICH Puthsreyneath</li>"
        f"</ol>"
        f"</div>",
        unsafe_allow_html=True,
    )
    main()
