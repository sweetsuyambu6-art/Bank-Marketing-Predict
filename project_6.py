
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import shap
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.title("Banking Marketing")

df=pd.read_csv(r"C:\Users\velku\Downloads\New folder\DS\project\project_6\claend_bank.csv")

col1,col2=st.columns(2)

with col1:
# metric Number of Days Paid
    st.header("Number of Days Paid")
    df=df["day_of_week"].sum()        
    st.metric(label="per year", value=f"Days {df}")
    df=pd.read_csv(r"C:\Users\velku\Downloads\New folder\DS\project\project_6\claend_bank.csv")
# metric Total Balance
    st.header("Total Balance")
    short=df["balance"].sum()        
    st.subheader(f"₹{short}")
# metric Toal Duration
    st.header("Toal Duration")    
    total_duration = df["duration"].sum()        
    st.subheader(total_duration)

with col2:   
# average age and df
    st.subheader("Average Age Marital")
    company = st.selectbox("Select Marital",df['marital'].unique(),key="marital")
    filtered_df = df[df['marital'] == company]
    close_price = df['age'].mean()  
    df=filtered_df[['age', 'marital',"housing","loan","duration"]].head(5)
    st.table(df)


#paying as per week st.bar chart
df=pd.read_csv(r"C:\Users\velku\Downloads\New folder\DS\project\project_6\all_togather.csv")
st.subheader("Paying As Per Days Of Week")
figure = st.bar_chart(df, x='month', y='day_of_week',color="loan", horizontal=True)

# average age and df
st.subheader("Contacted Loan And Balance")
company = st.selectbox("Select education",df['education'].unique(),key="education")
filtered_df = df[df['education'] == company]  
close_price = df['balance'].mean()  
df=filtered_df[['education', 'balance',"contact","loan"]].head()
st.table(df)

#age group bar chart
df=pd.read_csv(r"C:\Users\velku\Downloads\New folder\DS\project\project_6\all_togather.csv")
st.subheader("Group of Job and Bank Balance")
figure = px.bar(df,x="job",y="balance",text_auto=True)
st.plotly_chart(figure)

# data-input 
df=pd.read_csv(r"C:\Users\velku\Downloads\New folder\DS\project\project_6\all_togather.csv")
df.drop(["Unnamed: 0"],axis=1,inplace=True)
# traget 
x=df.drop("target",axis=1)
y=df["target"]

# one-hot-encoding 
x =pd.get_dummies(df, dtype=int)

# label-encoding
label = LabelEncoder()
y = label.fit_transform(y)

# train-test-data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x=scaler.fit_transform(x)

# Baseline and Advanced simpler models
models = [
    LogisticRegression(class_weight="balanced", max_iter=10000),
    RandomForestClassifier(random_state=42),
    XGBClassifier(),
    SVC(),
    KNeighborsClassifier(n_neighbors=2, weights="uniform", leaf_size=5),
    DecisionTreeClassifier(random_state=42),
    CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0),
]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Stratified K-Fold setup
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# model loop  
for model in models:
    results = cross_val_score(
        model,
        x_train,
        y_train,
        cv=skf
    )
# model fit-in
model.fit(x_train, y_train)
train_pred=model.predict(x_train)
train_pred_prob=model.predict_proba(x_train)

# compute SHAP values
st.header("Specific Prediction chart")
explainer = shap.Explainer(model)
shap_values = explainer(x_test)
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0])
st.pyplot(fig)











