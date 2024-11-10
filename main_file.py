import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

# Load data
df = pd.read_csv('user_behavior_dataset.csv')

# Function to change data type
def change_data_type(data, df_col, types):
    data[df_col] = data[df_col].astype(types)
    return data[df_col]

# Separate features and target
X = df.drop(columns=['User ID', 'Device Model', 'User Behavior Class'])
y = df['User Behavior Class']

# Encode 'Gender' as binary
X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1})
X['Operating System'] = X['Operating System'].map({'iOS': 0, 'Android': 1})

# Convert all columns to float type
for i in X.columns:
    change_data_type(X, i, float)

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=23)

# 1: Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# 2: Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=23)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# 3: Train SVM model
svm_model = SVC(kernel='rbf',probability=True, random_state=23)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# 4: Train Logistic Regression model using OneVsRestClassifier
log_reg_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=23))
log_reg_model.fit(X_train, y_train)
y_pred_log = log_reg_model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

# 5: Train MLP Neural Network model
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=23)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

# Prediction function
def predict_user_behaviour(new_data, model):
    new_data_df = pd.DataFrame([new_data])
    
    if 'Gender' in new_data_df:
        new_data_df['Gender'] = new_data_df['Gender'].map({'Female': 0, 'Male': 1})
        
    if 'Operating System' in new_data_df:
        new_data_df['Operating System'] = new_data_df['Operating System'].map({'iOS': 0, 'Android': 1})
        
        
    for i in new_data_df.columns:
        new_data_df[i] = new_data_df[i].astype(float)
        
    # Scale the new data
    new_data_scaled = scaler.transform(new_data_df)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)
    
    return prediction[0]

def predict_user_behaviour_pie(new_data_pie, model_pie):
    new_data_df_pie = pd.DataFrame([new_data_pie])
    new_data_df_pie['Gender'] = new_data_df_pie['Gender'].map({'Female': 0, 'Male': 1})
    new_data_df_pie['Operating System'] = new_data_df_pie['Operating System'].map({'iOS': 0, 'Android': 1})
    new_data_scaled = scaler.transform(new_data_df_pie.astype(float))
    return model_pie.predict_proba(new_data_scaled)[0]

def pie_chart(prediction_probs, name):
    class_labels = [f"Class {i+1}" for i in range(len(prediction_probs))]

    fig = go.Figure(
        data=[go.Pie(
            labels=class_labels,
            values=prediction_probs,
            hole=0.3,
            textinfo='label+percent',
            insidetextorientation='radial',
        )]
    )

    fig.update_layout(
        title_text=f"Predicted User Behavior Class Probability {name}",
        height=600,
        width=800
    )

    st.plotly_chart(fig)
    
st.set_page_config(page_title="Behavior analysis", layout="wide")

df_data = pd.read_csv('user_behavior_dataset.csv')
st.dataframe(df_data)

st.header("This is the Behavior analysis app.")

operating_sys_use_inp = st.sidebar.selectbox('Operating System', ['Android', 'iOS'])
app_use_inp = st.sidebar.number_input('App Usage Time (min/day):', min_value=0, value=120)
screen_use_inp = st.sidebar.number_input('Screen On Time (hours/day):', min_value=0, value=2)
battry_use_inp = st.sidebar.number_input('Battery Drain (mAh/day):', min_value=0, value=500)
num_app_use_inp = st.sidebar.number_input('Number of Apps Installed:', min_value=0, value=60)
data_use_inp = st.sidebar.number_input('Data Usage (MB/day):', min_value=0, value=600)
age_use_inp = st.sidebar.number_input('Age:', min_value=0, value=25)
gender_use_inp = st.sidebar.radio('Gender', ['Male', 'Female'])

new_user_data = {
    'Operating System': operating_sys_use_inp,
    'App Usage Time (min/day)': app_use_inp,
    'Screen On Time (hours/day)': screen_use_inp,
    'Battery Drain (mAh/day)': battry_use_inp,
    'Number of Apps Installed': num_app_use_inp,
    'Data Usage (MB/day)': data_use_inp,
    'Age': age_use_inp,
    'Gender': gender_use_inp,
}



show_models = st.selectbox('Select model to predict Behaviour: ', ['KNN', 'Random Forest','SVM', 'Logistic Regression','MLP','All Models'])

if show_models == 'KNN':
    st.markdown(f"Predicted User Behavior Class (KNN) : {predict_user_behaviour(new_user_data, knn)}")
    st.markdown(f"Accuracy (KNN) : {accuracy_knn * 100:.2f}%")
    prediction_probs = predict_user_behaviour_pie(new_user_data, knn)
    pie_chart(prediction_probs,show_models)

if show_models == 'Random Forest':
    st.markdown(f"Predicted User Behavior Class (Random Forest) : {predict_user_behaviour(new_user_data, rf_model)}")
    st.markdown(f"Accuracy (Random Forest) : {accuracy_rf * 100:.2f}%")
    prediction_probs = predict_user_behaviour_pie(new_user_data, rf_model)
    pie_chart(prediction_probs,show_models)

if show_models == 'SVM':
    st.markdown(f"Predicted User Behavior Class (SVM) : {predict_user_behaviour(new_user_data, svm_model)}")
    st.markdown(f"Accuracy (SVM) : {accuracy_svm * 100:.2f}%")
    prediction_probs = predict_user_behaviour_pie(new_user_data, svm_model)
    pie_chart(prediction_probs,show_models)

if show_models == 'Logistic Regression':
    st.markdown(f"Predicted User Behavior Class (Logistic Regression) :  {predict_user_behaviour(new_user_data, log_reg_model)}")
    st.markdown(f"Accuracy (Logistic Regression) : {accuracy_log * 100:.2f}%")
    prediction_probs = predict_user_behaviour_pie(new_user_data, log_reg_model)
    pie_chart(prediction_probs,show_models)

if show_models == 'MLP':
    st.markdown(f" Predicted User Behavior Class (MLP) : { predict_user_behaviour(new_user_data, mlp_model)}")
    st.markdown(f" Accuracy (MLP) : {accuracy_mlp* 100:.2f}%")  
    prediction_probs = predict_user_behaviour_pie(new_user_data, mlp_model)
    pie_chart(prediction_probs,show_models)

if show_models == 'All Models':
    st.markdown(f"1) Predicted User Behavior Class (KNN) : {predict_user_behaviour(new_user_data, knn)}")
    st.markdown(f"1) Accuracy (KNN) : {accuracy_knn * 100:.2f}%")
    
    st.markdown(f"2) Predicted User Behavior Class (Random Forest) : {predict_user_behaviour(new_user_data, rf_model)}")
    st.markdown(f"2) Accuracy (Random Forest) : {accuracy_rf * 100:.2f}%")

    st.markdown(f"3) Predicted User Behavior Class (SVM) : {predict_user_behaviour(new_user_data, svm_model)}")
    st.markdown(f"3) Accuracy (SVM) : {accuracy_svm * 100:.2f}%")
    
    st.markdown(f"4) Predicted User Behavior Class (Logistic Regression) :  {predict_user_behaviour(new_user_data, log_reg_model)}")
    st.markdown(f"4) Accuracy (Logistic Regression) : {accuracy_log * 100:.2f}%")
    
    st.markdown(f"5) Predicted User Behavior Class (MLP) : { predict_user_behaviour(new_user_data, mlp_model)}")
    st.markdown(f"5) Accuracy (MLP) : {accuracy_mlp* 100:.2f}%") 
    
    models = [knn, rf_model, svm_model, log_reg_model, mlp_model]  # List of trained models
    model_names = ["KNN", "Random Forest", "SVM", "Logistic Regression", "MLP"]
  # List to store prediction probabilities for each model

    for model,name in  zip(models, model_names):
        # Get the prediction probabilities using the predict_proba function
        prediction_probs = predict_user_behaviour_pie(new_user_data, model)
        # Generate a pie chart with the predicted probabilities
        pie_chart(prediction_probs, name)    

# Create the models and predictions list

models = ['KNN', 'Random Forest', 'SVM', 'Logistic Regression', 'MLP']
predictions = [y_pred_knn, y_pred_rf, y_pred_svm, y_pred_log, y_pred_mlp]

import random

# List of colors
colors = ['red', 'blue', 'purple','aliceblue', 'darkgreen', 'darkred', 'brown', 'darkgoldenrod', 'darkolivegreen']

# Randomly select a color
selected_color = random.choice(colors)
if 'visualization_started' not in st.session_state:
    st.session_state.visualization_started = False
    

if st.button('Visualize:'):
    st.session_state.visualization_started = True  # Set to True when button is pressed

if st.session_state.visualization_started:
    visual_inp = st.selectbox('Select Visualization Type:', ['None', 'Scatter Plot', 'Bar Chart'])

# Loop through each model and create a separate plot
    if visual_inp == 'Scatter Plot':
        for idx, model in enumerate(models):
            pred = predictions[idx]

            # Create a plot for Actual vs Predicted values
            fig = go.Figure()

            # Create a trace for Actual values
            trace_actual = go.Scatter(
                x=list(range(len(y_test))),
                y=y_test,
                mode='markers',
                marker=dict(color=f'{selected_color}', size=10, opacity=1),  
                name=f"Actual ({model})",
                line=dict(color='blue'),
                hoverinfo='x+y'
            )

            # Create a trace for Predicted values
            trace_pred = go.Scatter(
                x=list(range(len(pred))),
                y=pred,
                mode='markers',
                name=f"Predicted ({model})",
                line=dict(color='Yellow', dash='dash'),
                hoverinfo='x+y'
            )

            # Add traces to the figure
            fig.add_trace(trace_actual)
            fig.add_trace(trace_pred)

            # Update layout for the graph
            fig.update_layout(
                title=f"Actual vs Predicted ({model})",
                height=600,
                width=900,
                showlegend=True,
                hovermode='closest'
            )

            # Display the plot for the model
            st.plotly_chart(fig)

    elif visual_inp == 'Bar Chart':
        for idx, model in enumerate(models):
            pred = predictions[idx]

            # Create a bar chart for Actual vs Predicted values
            fig = go.Figure()

            # Bar for Actual values
            fig.add_trace(go.Bar(
                x=list(range(len(y_test))),
                y=y_test,
                name=f"Actual ({model})",
                marker_color='Blue' , # Use the selected background color for actual values
                width=1 
            ))

            # Bar for Predicted values
            fig.add_trace(go.Bar(
                x=list(range(len(pred))),
                y=pred,
                name=f"Predicted ({model})",
                marker_color='yellow'  # Use yellow for predicted values
            ))

            # Update layout for the bar chart
            fig.update_layout(
                title=f"Actual vs Predicted ({model})",
                barmode='group',  # Group bars side by side
                height=600,
                width=2900,
                showlegend=True,
                hovermode='closest'
            )

            # Display the bar chart for the model
            st.plotly_chart(fig)



st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: #888888; /* Change this color as desired */
        font-size: 14px;
    }
    </style>
    <div class="footer">
        
    </div>
    <div class="footer" style="right: 10px;">
        Developed by Faraz.
    </div>
    """,
    unsafe_allow_html=True
)
