import gc
import tempfile
import os
import random
import time
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
import streamlit as st
import pandas as pd
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Logging configuration
logging.basicConfig(filename="debug.log", level=logging.DEBUG)

def SystemPrint(message):
    logging.debug(f"System: {message}")
    print(f"System: {message}")

# Clean temporary directory
def CleanTempDir():
    tempDir = tempfile.gettempdir()
    for root, dirs, files in os.walk(tempDir):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except Exception as e:
                continue

CleanTempDir()

# Main Functions
class MainFunctions:
    @staticmethod
    def Average(numList, rateList):
        total = 0
        for it, r in zip(numList, rateList):
            total += it * r
        return total / sum(rateList)

    @staticmethod
    def RedefineLabel(percentage):
        return "Scam" if percentage >= 50 else "Normal"

    @staticmethod
    def get_prediction_proba(classifier, question_tfidf):
        prediction_proba = classifier.predict_proba(question_tfidf)[0]
        spam_proba = prediction_proba[1] * 100
        ham_proba = prediction_proba[0] * 100
        return spam_proba, ham_proba

    @staticmethod
    def AskingQuestion(question):
        err = 1
        while err == 1:
            try:
                result = st.session_state.chat.send_message(question)
                err = 0
            except Exception as e:
                SystemPrint(f"Error: {e}")
        return result.text

    @staticmethod
    def stringIsNullOrBlank(string):
        return string == "" or string.isspace()

    @staticmethod
    def stringStartWithlnOrBlank(string):
        return string.startswith("\n") or string.startswith(" ")

    @staticmethod
    def stringEndWithlnOrBlank(string):
        return string.endswith("\n") or string.endswith(" ")

# Configure Streamlit page
st.set_page_config(layout="wide")
st.title("詐騙簡訊偵測器 Anti-Scam Tester")
st.subheader("模型測試精確度 Model Accuracy")

# Gemini configuration
genai.configure(api_key="AIzaSyBflj_zpaKbeFyv9WkOVM3d4iJVb5Vz2Hk")
model = genai.GenerativeModel("gemini-2.0-flash-exp")
st.session_state.chat = model.start_chat(history=[])

# Cache vectorizer and models
@st.cache_resource
def load_vectorizer_and_models():
    vectorizer = TfidfVectorizer()
    classifiers = {
        "LRclassifier": LogisticRegression(n_jobs=-1),
        "SVCclassifier": CalibratedClassifierCV(LinearSVC(dual=False), n_jobs=-1),
        "NBclassifier": MultinomialNB(alpha=0.1, fit_prior=True),
        "SGDclassifier": CalibratedClassifierCV(SGDClassifier(n_jobs=-1, loss='hinge')),
        "STACKclassifier": StackingClassifier(estimators=[
            ('lr', LogisticRegression(n_jobs=-1)),
            ('svc', CalibratedClassifierCV(LinearSVC(dual=False), n_jobs=-1)),
            ('nb', MultinomialNB(alpha=0.1, fit_prior=True)),
            ('sgd', CalibratedClassifierCV(SGDClassifier(n_jobs=-1, loss='hinge')))
        ], final_estimator=LogisticRegression(), n_jobs=-1)
    }
    return vectorizer, classifiers

vectorizer, classifiers = load_vectorizer_and_models()

# Training function
def train_models():
    try:
        with st.spinner("正在載入資料... Loading data..."):
            labels, messages = [], []
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")
            with open(file_path, encoding="utf-8") as file:
                dataList = file.read().split('\n')

            for i, line in enumerate(dataList):
                try:
                    if not line.strip():
                        continue
                    label, message = line.split('\t')
                    labels.append(1 if label == 'spam' else 0)
                    messages.append(message)
                except Exception as e:
                    SystemPrint(f"Error in line {i+1}: {line} - {e}")
                    exit(-1)

            X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=random.randint(0, 114514))
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            st.session_state.update({
                "Xtfidf": X_test_tfidf,
                "Ytfidf": y_test
            })

        with st.spinner("正在訓練模型... Training models..."):
            for name, classifier in classifiers.items():
                classifier.fit(X_train_tfidf, y_train)

            st.session_state.update({
                "vectorizer": vectorizer,
                "classifiers": classifiers
            })

    except Exception as e:
        SystemPrint(f"Training failed. Reason: {e}")

if 'modelTrained' not in st.session_state:
    train_models()
    st.session_state.modelTrained = True

# Main application logic
def main():
    try:
        with st.spinner("正在測試模型... Testing models..."):
            accuracy_data = []
            for model_name, classifierKey in classifiers.items():
                accuracy = accuracy_score(st.session_state.Ytfidf, classifierKey.predict(st.session_state.Xtfidf)) * 100
                accuracy_data.append({"模型 Model": model_name, "準確度 Accuracy": f"{accuracy:.2f}%"})

        st.table(pd.DataFrame(accuracy_data))

        message = st.text_area("輸入要測試的訊息：\nEnter your message to analyze:", height=200)
        if st.button("分析訊息 Analyze Message"):
            if not message or message.isspace():
                st.warning("請先輸入訊息。Please enter a message to analyze.")
                st.stop()

            with st.spinner("正在分析訊息... Analyzing message..."):
                translation = MainFunctions.AskingQuestion(f"""
                    Is this passage in ALL English? If so, return the original passage. If not, translate ALL non-English parts to English and return the new passage.
                    ONLY RETURN THE RESULT OF the original message or the translated message. DO NOT ADD OTHER WORDS!!!
                    message: {message}""")
                AiJudgement = MainFunctions.AskingQuestion(f"""
                    How much percentage do you think this message is a spamming message?
                    Answer me in this format: "N"
                    For N is a float between 0~100.
                    message: {translation}""")
                AiJudgePercentage = float(AiJudgement)

                question_tfidf = vectorizer.transform([translation])
                results_data = []

                for model_name, classifier in classifiers.items():
                    spam_proba, ham_proba = MainFunctions.get_prediction_proba(classifier, question_tfidf)
                    results_data.append({
                        "模型 Model": model_name,
                        "結果 Result": MainFunctions.RedefineLabel(spam_proba),
                        "詐騙訊息機率 Scam Probability": f"{spam_proba:.2f}%",
                        "普通訊息機率 Normal Probability": f"{ham_proba:.2f}%"
                    })

                results_data.append({
                    "模型 Model": "Gemini",
                    "結果 Result": MainFunctions.RedefineLabel(AiJudgePercentage),
                    "詐騙訊息機率 Scam Probability": f"{AiJudgePercentage:.2f}%",
                    "普通訊息機率 Normal Probability": f"{100.0 - AiJudgePercentage:.2f}%"
                })

                st.subheader("分析結果 Analysis Results")
                st.dataframe(pd.DataFrame(results_data))

    except Exception as e:
        SystemPrint(f"Error! Reason: {e}")
        input("System: Press any key to exit...")

if __name__ == "__main__":
    main()
