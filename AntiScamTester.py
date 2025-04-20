import os, warnings, asyncio, time
import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from googletrans import Translator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
def show_error(error_message):
    st.error(f"執行時錯誤 Error: {error_message}")

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
        try:
            result = st.session_state.chat.send_message(question)
        except Exception as e:
            show_error(f"Gemini處理失敗...Gemini processing failed. 原因 Reason: {e}")

        return result.text
    
    @staticmethod
    async def Translate(translator: Translator, message, source_language='auto', target_language='en'):
        translated = await translator.translate(message, src=source_language, dest=target_language)
        return translated.text

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models and configuration
vectorizer = TfidfVectorizer()
LRclassifier = LogisticRegression(n_jobs=-1)
SVCclassifier = CalibratedClassifierCV(LinearSVC(dual=False), n_jobs=-1)
NBclassifier = MultinomialNB(alpha=0.08451, fit_prior=True)
SGDclassifier = CalibratedClassifierCV(SGDClassifier(n_jobs=-1, loss='hinge'))
DTclassifier = DecisionTreeClassifier()
RFclassifier = RandomForestClassifier(n_jobs=-1)
STACKclassifier = StackingClassifier(estimators=[
    ('lr', LRclassifier), 
    ('svc', SVCclassifier), 
    ('nb', NBclassifier), 
    ('sgd', SGDclassifier),
    ('dt', DTclassifier),
    ('rf', RFclassifier)
], final_estimator=LogisticRegression(), n_jobs=-1)

models = [
    ("sklearn (邏輯迴歸 Logistic Regression)", "LRclassifier", 1),
    ("sklearn (支援向量機 Support Vector Classification)", "SVCclassifier", 1),
    ("sklearn (單純貝氏 Naive Bayes)", "NBclassifier", 1),
    ("sklearn (隨機梯度下降 Stochastic Gradient Descent)", "SGDclassifier", 1),
    ("sklearn (決策樹 Decision Tree)", "DTclassifier", 1),
    ("sklearn (隨機森林 Random Forest)", "RFclassifier", 1),
    ("sklearn (堆疊 Stacking)", "STACKclassifier", 1)
]

st.set_page_config(layout="wide")
st.title("詐騙簡訊偵測器 Anti-Scam Tester")
st.subheader("模型測試精確度 Model Accuracy")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")
st.session_state.chat = model.start_chat(history=[])

@st.cache_resource(show_spinner=False)
def load_and_train_models():
    try:
        num_models_to_train = len(models) # 使用全域 models 列表計算數量
        total_steps = 2 + num_models_to_train
        current_step = 0
        progress_text = "正在初始化模型載入... Initializing model loading..."
        progress_bar = st.progress(0, text=progress_text)

        def update_progress(step_increment, text):
            nonlocal current_step
            current_step += step_increment
            progress_percentage = min(1.0, current_step / total_steps)
            progress_bar.progress(progress_percentage, text=text)
            time.sleep(0.1)

        update_progress(0, "正在讀取訓練資料... Reading training data...") # 更新初始文字
        labels, messages = [], []
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")
        with open(file_path, encoding="utf-8") as f:
            dataList = f.read().split('\n')
            for i, line in enumerate(dataList):
                if not line.strip():
                    continue

                try:
                    label, message = line.split('\t')
                    labels.append(1 if label == 'spam' else 0)
                    messages.append(message)
                except ValueError as e:
                    show_error(f"訓練集有誤...An error occurred in the training data. 位於行 Position: {i+1}: {line} - {e}")
                    continue
        update_progress(1, "資料讀取完成. Data reading complete.") # 完成第 1 步

        update_progress(0, "正在分割與向量化資料... Splitting and vectorizing data...")
        X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        update_progress(1, "資料向量化完成. Data vectorization complete.") # 完成第 2 步

        classifiers = {
            "LRclassifier": LRclassifier,
            "SVCclassifier": SVCclassifier,
            "NBclassifier": NBclassifier,
            "SGDclassifier": SGDclassifier,
            "STACKclassifier": STACKclassifier,
            "DTclassifier": DTclassifier,
            "RFclassifier": RFclassifier
        }

        i = 1
        for name, classifier in classifiers.items():
            model_display_name = models[i-1][0]
            update_progress(0, f"正在訓練模型 ({i}/{num_models_to_train}): {model_display_name}...")
            classifier.fit(X_train_tfidf, y_train)
            update_progress(1, f"模型訓練完成: {model_display_name}.")
            i = i + 1

        progress_bar.progress(1.0, text="所有模型載入與訓練完成. All models loaded and trained.")
        time.sleep(0.5) # 短暫停留讓使用者看到完成狀態
        progress_bar.empty() # 完成後移除進度條

        return classifiers, X_test_tfidf, y_test, vectorizer
    except Exception as e:
        show_error(f"訓練失敗...Training failed. 原因 Reason: {e}")
        return None, None, None, None

if 'models' not in st.session_state:
    st.session_state.classifiers, st.session_state.Xtfidf, st.session_state.Ytfidf, st.session_state.vectorizer = load_and_train_models()
    st.session_state.models = models
    st.session_state.modelTrained = True
    st.session_state.translator = Translator()

def main():
    try:
        accuracy_data = []
        for model_name, classifierKey, _ in st.session_state.models:
            test_classifier = st.session_state.classifiers[classifierKey]
            accuracy = accuracy_score(st.session_state.Ytfidf, test_classifier.predict(st.session_state.Xtfidf)) * 100
            recall = recall_score(st.session_state.Ytfidf, test_classifier.predict(st.session_state.Xtfidf)) * 100
            precision = precision_score(st.session_state.Ytfidf, test_classifier.predict(st.session_state.Xtfidf)) * 100
            accuracy_data.append({
                "模型 Model": model_name, 
                "準確度 Accuracy": f"{accuracy:.2f}%",
                "召回率 Recall": f"{recall:.2f}%",
                "精準度 Precision": f"{precision:.2f}%"
            })
        st.table(pd.DataFrame(accuracy_data))
        st.markdown("""
        **指標說明 (Metric Explanations):**

        *   **準確度 (Accuracy):**
            *   模型整體預測正確的比例（包含正確預測為詐騙和正確預測為普通的樣本）。
            *   *Accuracy: The overall proportion of correct predictions (both scam and normal).*

        *   **召回率 (Recall):**
            *   在所有 **實際為詐騙** 的訊息中，模型成功將其預測為詐騙的比例。高召回率表示模型擅長找出詐騙訊息，較少漏網之魚。
            *   *Recall: The proportion of **actual scam** messages that the model correctly identified as scam. High recall means the model is good at finding most of the scam messages.*

        *   **精準度 (Precision):**
            *   在所有 **模型預測為詐騙** 的訊息中，實際真的是詐騙的比例。高精準度表示模型預測為詐騙的訊息可信度高，較少誤判（將普通訊息判為詐騙）。
            *   *Precision: The proportion of messages **predicted as scam** by the model that are actually scam. High precision means that when the model predicts scam, it is very likely to be correct.*
        """)

        message = st.text_area("輸入要測試的訊息：\nEnter your message to analyze:", height=200).strip()
        if st.button("分析訊息 Analyze Message"):
            if not message or message.isspace():
                st.warning("請先輸入訊息。Please enter a message to analyze.")
                st.stop()
            
            if 'last_message' not in st.session_state:
                st.session_state.last_message = ""

            if message == st.session_state.last_message:
                st.warning("與上一則訊息重複。This message is a duplicate of the previous message.")
                st.stop()

            st.session_state.last_message = message
            print(f"Last message: {st.session_state.last_message}")
            with st.spinner("正在分析訊息... Analyzing message..."):
                # Translation and AI Judgement
                translation = asyncio.run(MainFunctions.Translate(st.session_state.translator, message))
                
                AiJudgement = MainFunctions.AskingQuestion(f"""How much percentage do you think this message is a spamming message (only consider this message, not considering other environmental variation)? 
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
        st.session_state.last_message = ""
        information = [
            "如果下方警告訊息為\"Event loop is closed\", 只需再按一次按鈕即可。",
            "If the error message below is \"Event loop is closed\", just click the button again."
        ]

        st.toast("\n".join(information), icon="ℹ️")
        show_error(f"原因 Reason: {e}")

if __name__ == "__main__":
    main()
