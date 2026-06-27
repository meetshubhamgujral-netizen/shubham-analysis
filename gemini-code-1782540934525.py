import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Universal Data Analyzer", layout="wide", page_icon="📊")

# Initialize Gemini API
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.sidebar.warning("⚠️ Please configure your GEMINI_API_KEY in Streamlit Advanced Settings -> Secrets.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI HEADER ---
st.title("📊 Universal Data Analytics Dashboard")
st.markdown("Upload any dataset to begin diagnostic analysis, machine learning modeling, and AI-powered Q&A.")

# --- FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is None:
    st.info("👈 Please upload a dataset in the sidebar to begin analysis.")
else:
    # Read Data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # --- MAIN TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Exploratory Analysis", "🤖 Diagnostic Models", "🧠 Gemini Chat Insights"])

    # ==========================================
    # TAB 1: DESCRIPTIVE ANALYSIS
    # ==========================================
    with tab1:
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isna().sum().sum())

        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Interactive Visualizations")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            if numeric_cols:
                hist_col = st.selectbox("Select column for Distribution:", numeric_cols)
                fig_hist = px.histogram(df, x=hist_col, template="plotly_white", color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig_hist, use_container_width=True)
                
        with viz_col2:
            if len(numeric_cols) > 1:
                scatter_x = st.selectbox("Scatter X-axis:", numeric_cols, index=0)
                scatter_y = st.selectbox("Scatter Y-axis:", numeric_cols, index=1)
                fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, template="plotly_white", color_discrete_sequence=['#EF553B'])
                st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
    # TAB 2: MACHINE LEARNING (Diagnostics)
    # ==========================================
    with tab2:
        st.header("Predictive Diagnostics")
        st.write("Select a categorical target variable to train models and view ROC curves.")
        
        target_col = st.selectbox("Select Target Variable:", df.columns)
        
        if st.button("Train Models & Generate ROC"):
            with st.spinner("Training models..."):
                df_ml = df.dropna().copy()
                
                # Label encode target
                le = LabelEncoder()
                if df_ml[target_col].dtype == 'object':
                    df_ml[target_col] = le.fit_transform(df_ml[target_col])

                X = df_ml.drop(columns=[target_col])
                y = df_ml[target_col]

                # Convert categoricals to dummies
                X = pd.get_dummies(X, drop_first=True)

                if len(y.unique()) != 2:
                    st.warning("ROC Curves require a binary target (only 2 unique values). Showing accuracy only.")
                    can_plot_roc = False
                else:
                    can_plot_roc = True

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                models = {
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier()
                }

                fig_roc = go.Figure()
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.subheader("Model Accuracy")
                    for name, clf in models.items():
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        st.write(f"**{name}:** {acc:.2f}")

                        if can_plot_roc:
                            y_score = clf.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_score)
                            roc_auc = auc(fpr, tpr)
                            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC = {roc_auc:.2f})", mode='lines'))

                with res_col2:
                    if can_plot_roc:
                        st.subheader("ROC Curves")
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='gray'), name='Random'))
                        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', template="plotly_dark")
                        st.plotly_chart(fig_roc, use_container_width=True)

    # ==========================================
    # TAB 3: GEMINI AI CHAT
    # ==========================================
    with tab3:
        st.header("Ask Gemini About Your Data")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        data_context = f"""
        Dataset Summary:
        Columns: {df.columns.tolist()}
        Shape: {df.shape}
        Stats: {df.describe().to_markdown()}
        Please answer the user's analytical questions based on this dataset.
        """

        if prompt := st.chat_input("Ask a question about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    full_prompt = f"{data_context}\n\nUser Question: {prompt}"
                    response = model.generate_content(full_prompt)
                    message_placeholder.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"API Error: Make sure your new API key is in Streamlit Secrets. Details: {e}")
