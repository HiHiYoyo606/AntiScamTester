import tempfile, os, warnings
import google.generativeai as genai
import pandas as pd
import streamlit as st
from googletrans import Translator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Helper functions
def SystemPrint(message):
    print(f"System: {message}")

def CleanTempDir():
    tempDir = tempfile.gettempdir()
    for root, dirs, files in os.walk(tempDir):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except Exception as e:
                continue
CleanTempDir()

class MainFunctions:
    @staticmethod
    def Average(numList, rateList):
        total = 0
        for it, r in zip(numList, rateList):
            total += it * r
        return total / sum(rateList) 

    @staticmethod
    def RedefineLabel(percentage):
        return "詐騙 Scam" if percentage >= 50 else "普通 Normal"

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
    async def Translate(translator, message, source_language='auto', target_language='en'):
        translated = await translator.translate(text, src=source_language, dest=target_language)
        return translated.text

# Models and configuration
vectorizer = TfidfVectorizer()
LRclassifier = LogisticRegression(n_jobs=-1)
SVCclassifier = CalibratedClassifierCV(LinearSVC(dual=False), n_jobs=-1)
NBclassifier = MultinomialNB(alpha=0.08451, fit_prior=True)
SGDclassifier = CalibratedClassifierCV(SGDClassifier(n_jobs=-1, loss='hinge'))
STACKclassifier = StackingClassifier(estimators=[
    ('lr', LRclassifier), 
    ('svc', SVCclassifier), 
    ('nb', NBclassifier), 
    ('sgd', SGDclassifier)
], final_estimator=LogisticRegression(), n_jobs=-1)

models = [
    ("sklearn (邏輯迴歸 Logistic Regression)", "LRclassifier", 1),
    ("sklearn (支援向量機 Support Vector Classification)", "SVCclassifier", 1),
    ("sklearn (單純貝氏 Naive Bayes)", "NBclassifier", 1),
    ("sklearn (隨機梯度下降 Stochastic Gradient Descent)", "SGDclassifier", 1),
    ("sklearn (堆疊 Stacking)", "STACKclassifier", 1)
]

st.set_page_config(layout="wide")
st.title("詐騙簡訊偵測器 Anti-Scam Tester")
st.subheader("模型測試精確度 Model Accuracy")

# Configure Gemini
genai.configure(api_key="AIzaSyBflj_zpaKbeFyv9WkOVM3d4iJVb5Vz2Hk")
model = genai.GenerativeModel("gemini-2.0-flash-exp")
st.session_state.chat = model.start_chat(history=[])

@st.cache_resource
def load_and_train_models():
    try:
        with st.spinner("正在載入資料... Loading data..."):
            labels, messages = [], []
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")
            with open(file_path, encoding="utf-8") as f:
                dataList = f.read().split('\n')
                for i, line in enumerate(dataList):
                    if line.strip():
                        try:
                            label, message = line.split('\t')
                            labels.append(1 if label == 'spam' else 0)
                            messages.append(message)
                        except ValueError as e:
                            SystemPrint(f"Error in line {i+1}: {line} - {e}")
                            continue

            X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            classifiers = {
                "LRclassifier": LRclassifier,
                "SVCclassifier": SVCclassifier,
                "NBclassifier": NBclassifier,
                "SGDclassifier": SGDclassifier,
                "STACKclassifier": STACKclassifier
            }

            for name, classifier in classifiers.items():
                classifier.fit(X_train_tfidf, y_train)

            return classifiers, X_test_tfidf, y_test, vectorizer
    except Exception as e:
        SystemPrint(f"Training failed. Reason: {e}")
        return None, None, None, None

if 'models' not in st.session_state:
    st.session_state.classifiers, st.session_state.Xtfidf, st.session_state.Ytfidf, st.session_state.vectorizer = load_and_train_models()
    st.session_state.models = models
    st.session_state.modelTrained = True
    st.session_state.translator = Translator()

async def main():
    try:
        with st.spinner("正在測試模型... Testing models..."):
            accuracy_data = []
            for model_name, classifierKey, _ in st.session_state.models:
                test_classifier = st.session_state.classifiers[classifierKey]
                accuracy = accuracy_score(st.session_state.Ytfidf, test_classifier.predict(st.session_state.Xtfidf)) * 100
                accuracy_data.append({"模型 Model": model_name, "準確度 Accuracy": f"{accuracy:.2f}%"})
        st.table(pd.DataFrame(accuracy_data))

        message = st.text_area("輸入要測試的訊息：\nEnter your message to analyze:", height=200)
        if st.button("分析訊息 Analyze Message"):
            if not message or message.isspace():
                st.warning("請先輸入訊息。Please enter a message to analyze.")
                st.stop()

            with st.spinner("正在分析訊息... Analyzing message..."):
                # Translation and AI Judgement
                translation = await MainFunctions.Translate(st.session_state.translator, message=message)

                AiJudgement = MainFunctions.AskingQuestion(f"""How much percentage do you think this message is a spamming message? 
                    Answer in this format: "N" where N is a float between 0-100 (13.62, 85.72, 50.60, 5.67, 100.00, 0.00 etc.)
                    message: {translation}""")

                AiJudgePercentage = float(AiJudgement)
                AiJudgePercentageRate = 1

                # Model Analysis
                question_tfidf = st.session_state.vectorizer.transform([translation])
                results_data, result_row = [], []

                for model_name, classifierKey, rate in models:
                    working_classifier = st.session_state.classifiers[classifierKey]
                    spam_proba, ham_proba = MainFunctions.get_prediction_proba(working_classifier, question_tfidf)
                    results_data.append({
                        "模型 Model": model_name,
                        "結果 Result": MainFunctions.RedefineLabel(spam_proba),
                        "加權倍率 Rate": rate,
                        "詐騙訊息機率 Scam Probability": f"{spam_proba:.2f}%",
                        "普通訊息機率 Normal Probability": f"{ham_proba:.2f}%"
                    })

                # Add Gemini results
                results_data.append({
                    "模型 Model": "Google Gemini",
                    "結果 Result": MainFunctions.RedefineLabel(AiJudgePercentage),
                    "加權倍率 Rate": AiJudgePercentageRate,
                    "詐騙訊息機率 Scam Probability": f"{AiJudgePercentage:.2f}%",
                    "普通訊息機率 Normal Probability": f"{100.0 - AiJudgePercentage:.2f}%"
                })

                # Calculate final result
                spam_percentages = [float(d["詐騙訊息機率 Scam Probability"].rstrip('%')) for d in results_data]
                rates = [d["加權倍率 Rate"] for d in results_data]
                final_spam_percentage = MainFunctions.Average(spam_percentages, rates)
                final_ham_percentage = 100.0 - final_spam_percentage

                # Add final result
                result_row.append({
                    "模型 Model": "加權平均分析結果 Weighted Average Analysis Result",
                    "結果 Result": MainFunctions.RedefineLabel(final_spam_percentage),
                    "加權倍率 Rate": sum(rates),
                    "詐騙訊息機率 Scam Probability": f"{final_spam_percentage:.2f}%",
                    "普通訊息機率 Normal Probability": f"{final_ham_percentage:.2f}%"
                })
                
                # Highlight the last row based on result
                def highlight_row(row):
                    bgcolor = "lightgreen" if row["結果 Result"] == MainFunctions.RedefineLabel(0) else "lightcoral"
                    fontcolor = "black"
                    return [f"background-color: {bgcolor}; color: {fontcolor}; font-weight: 700"] * len(row)

                # Display results with final row highlight
                rddf, rrdf = pd.DataFrame(results_data), pd.DataFrame(result_row)
                st.subheader("個別分析結果 Individual Analysis Results")
                st.dataframe(rddf.style.apply(highlight_row, axis=1))
                st.subheader("綜合分析結果 Comprehensive Analysis Result")
                st.dataframe(rrdf.style.apply(highlight_row, axis=1))
                st.subheader("翻譯紀錄 Translation Log")
                st.text_area("", value=translation, height=200, disabled=True)

    except Exception as e:
        SystemPrint(f"Error! Reason:{e}")
        input("System: Press any key to exit. . .")

if __name__ == "__main__":
    main()
