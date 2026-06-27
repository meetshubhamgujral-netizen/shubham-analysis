import sys
import subprocess

# --- FORCE AUTO-INSTALLER ---
# This bypasses the need for requirements.txt by installing missing packages on the fly
try:
    import plotly.express as px
    import sklearn
    import google.generativeai as genai
except ImportError:
    import streamlit as st
    st.warning("⏳ Installing missing libraries... Please wait 30 seconds, the app will automatically refresh!")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "scikit-learn", "google-generativeai", "openpyxl"])
    st.rerun()

# --- STANDARD IMPORTS ---
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
# (The rest of your existing code continues from here down...)
st.set_page_config(page_title="Universal Data Analyzer", layout="wide", page_icon="📊")
