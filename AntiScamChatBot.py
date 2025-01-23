def SystemPrint(message):
    print(f"System: {message}")

import gc, tempfile, os, random, time, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

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
    def Average(numList, rateList):
        total = 0
        for it, r in zip(numList, rateList):
            total += it*r
        
        return total / sum(rateList) 

    def RedefineLabel(percentage):
        return ("Scam" if percentage >= 50 else "Normal")

    def get_prediction_proba(classifier, question_tfidf):
        prediction_proba = classifier.predict_proba(question_tfidf)[0]
        spam_proba = prediction_proba[1] * 100
        ham_proba = prediction_proba[0] * 100
        return spam_proba, ham_proba
    
    def AskingQuestion(question):
            err = 1
            while err == 1:
                try:
                    result = st.session_state.chat.send_message(question)
                    err = 0
                except Exception as e:
                    SystemPrint(f"Error: {e}")
            return result.text
    
    def stringIsNullOrBlank(string):
        return string == "" or string.isspace()
    
    def stringStartWithlnOrBlank(string):
        return string.startswith("\n") or string.startswith(" ")
    
    def stringEndWithlnOrBlank(string):
        return string.endswith("\n") or string.endswith(" ")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
import google.generativeai as genai
import streamlit as st
import pandas as pd

#Models
vectorizer = TfidfVectorizer()
LRclassifier = LogisticRegression(n_jobs=-1)
SVCclassifier = CalibratedClassifierCV(LinearSVC(dual=False), n_jobs=-1)
NBclassifier = MultinomialNB(alpha=0.1, fit_prior=True)
SGDclassifier = CalibratedClassifierCV(SGDClassifier(n_jobs=-1, loss='hinge'))
STACKclassifier = StackingClassifier(estimators=[
    ('lr', LRclassifier), 
    ('svc', SVCclassifier), 
    ('nb', NBclassifier), 
    ('sgd', SGDclassifier)
], final_estimator=LogisticRegression(), n_jobs=-1)

models = [ #(model, classifier, rate)
    ("sklearn (邏輯迴歸 Logistic Regression)", "LRclassifier", 1),
    ("sklearn (支援向量機 Support Vector Classification)", "SVCclassifier", 1),
    ("sklearn (單純貝氏 Naive Bayes)", "NBclassifier", 1),
    ("sklearn (隨機梯度下降 Stochastic Gradient Descent)", "SGDclassifier", 1),
    ("sklearn (堆疊 Stacking)", "STACKclassifier", 1)
]

st.set_page_config(layout="wide")
st.title("詐騙簡訊偵測器 Anti-Scam Tester")
st.subheader("模型測試精確度 Model Accuracy")

# 配置Gemini
genai.configure(api_key="AIzaSyBflj_zpaKbeFyv9WkOVM3d4iJVb5Vz2Hk")
model =genai.GenerativeModel("gemini-2.0-flash-exp")
st.session_state.chat = model.start_chat(history=[])

def train_models():
    try:
        with st.spinner("正在載入資料... Loading data..."):
        #訓練資料
            labels, messages = [], []
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")
            file = open(file_path, encoding="utf-8").read()
            dataList = file.split('\n')
            step = 100/len(dataList)
            for i, line in enumerate(dataList):
                try:
                    if not line.strip(): 
                        continue

                    label, message = line.split('\t')
                    if label == 'spam':
                        labels.append(1)
                    elif label == 'ham':
                        labels.append(0) 
                    else:
                        raise ValueError(f"Unknown label: {label}")    
                    messages.append(message)
                except Exception as e:
                    SystemPrint(f"Error in line {i+1}: {line} - {e}")
                    exit(-1)
            X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=random.randint(0, 114514))
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            if 'Xtfidf' not in st.session_state:
                st.session_state.Xtfidf = X_test_tfidf
            if 'Ytfidf' not in st.session_state:
                st.session_state.Ytfidf = y_test

        with st.spinner("正在訓練模型... Training models..."):
            # 訓練模型
            classifiers = [{
                "LRclassifier": LRclassifier,
                "SVCclassifier": SVCclassifier,
                "NBclassifier": NBclassifier,
                "SGDclassifier": SGDclassifier,
                "STACKclassifier": STACKclassifier
            }]

            for name, classifier in classifiers[0].items():
                classifier.fit(X_train_tfidf, y_train)
            st.session_state.models = models
            st.session_state.vectorizer = vectorizer
            st.session_state.classifiers = classifiers
            st.session_state.lastText = "" ############################################## HERE
    except Exception as e:
        SystemPrint(f"Training falied. Reason: {e}")

def main():
    try:
        # 顯示準確度
        with st.spinner("正在測試模型... Testing models..."):
            accuracy_data = []
            for model_name, classifierKey, _ in st.session_state.models:
                test_classifier = st.session_state.classifiers[0][classifierKey]
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
                translation = MainFunctions.AskingQuestion(f"""
                    Is this passage in ALL English? If so, return the orignial passage. If not, traslate ALL non-English part to English and return the new passage.
                    ONLY RETURN THE RESULT OF original message or the translated message. DO NOT ADD OTHER WORDS!!!
                    message: {message}""")
                print(translation)
                
                time.sleep(1)
                AiJudgement = MainFunctions.AskingQuestion(f"""how much percentage do you think this message is a spamming message?
                    answer me for this format: \"N\"
                    for N is an float between 0~100. (13.62, 85.72, 50.60, 5.67, 100.00, 0.00 etc.)
                    message: {translation}""")
                
                AiJudgePercentage = float(AiJudgement)
                AiJudgePercentageRate = 1

                # Model Analysis
                question_tfidf = st.session_state.vectorizer.transform([translation])
                results_data = []

                for model_name, classifierKey, rate in models:
                    working_classifier = st.session_state.classifiers[0][classifierKey]
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
                    "模型 Model": "Gemini",
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
                results_data.append({
                    "模型 Model": "Final Result",
                    "結果 Result": MainFunctions.RedefineLabel(final_spam_percentage),
                    "加權倍率 Rate": sum(rates),
                    "詐騙訊息機率 Scam Probability": f"{final_spam_percentage:.2f}%",
                    "普通訊息機率 Normal Probability": f"{final_ham_percentage:.2f}%"
                })

                # Display results
                st.subheader("分析結果 Analysis Results")
                df = pd.DataFrame(results_data)
                
                # 为最后一行设置不同的背景色
                def highlight_final_row(row):
                    if row.name == len(df) - 1:  # 最后一行
                        bgcolor = "lightgreen" if row["結果 Result"] == 'Normal' else "lightcoral"
                        fontColor = "black"
                        color = f"background-color: {bgcolor}; color: {fontColor}; font-weight: bold"
                        return [color] * len(row)
                    return [''] * len(row)

                st.dataframe(df.style.apply(highlight_final_row, axis=1))

    except Exception as e:
        SystemPrint(f"Error! Reason:{e}")
        input("System: Press any key to exit. . .")

if __name__ == "__main__":
    main()
