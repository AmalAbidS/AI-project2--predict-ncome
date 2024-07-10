import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# Read the data into a pandas DataFrame
data = pd.read_csv("/content/adult.csv")

# Replace (?) with NaNs
data.replace('?', np.nan, inplace=True)

# Select relevant features
selected_features = ['age', 'marital-status', 'occupation', 'educational-num', 'hours-per-week']

# Split hours-per-week into fulltime and parttime
data['hours-per-week'] = pd.cut(data['hours-per-week'], bins=[0, 39, data['hours-per-week'].max()], labels=['parttime', 'fulltime'])

# Create feature matrix X and target vector y
X = data[selected_features]
y = data['income']  # Assuming 'income' is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = ['age', 'educational-num']
categorical_features = ['marital-status', 'occupation', 'hours-per-week']

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the pipeline for Random Forest Classifier
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline_rf.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline_rf, 'model.joblib')

# Evaluation
rf_predictions = pipeline_rf.predict(X_test)
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.joblib')

# Streamlit app
def main():
    st.title('Income')

    # Feature explanations
    st.write('Answer the questions to predict your salary')

    # Age slider
    age = st.slider('Select age:', min_value=17, max_value=90, value=30)

    # Marital status dropdown
    marital_status_options = {
        0: 'Never-married',
        1: 'Married-civ-spouse',
        2: 'Divorced',
        3: 'Separated',
        4: 'Widowed',
        5: 'Married-spouse-absent',
        6: 'Married-AF-spouse'
    }
    marital_status = st.selectbox('Select marital status:', options=list(marital_status_options.values()))

    # Occupation dropdown
    occupation_options = {
        0: 'Tech-support',
        1: 'Craft-repair',
        2: 'Other-service',
        3: 'Sales',
        4: 'Exec-managerial',
        5: 'Prof-specialty',
        6: 'Handlers-cleaners',
        7: 'Machine-op-inspct',
        8: 'Adm-clerical',
        9: 'Farming-fishing',
        10: 'Transport-moving',
        11: 'Priv-house-serv',
        12: 'Protective-serv',
        13: 'Armed-Forces'
    }
    occupation = st.selectbox('Select occupation:', options=list(occupation_options.values()))

    # Educational num slider
    educational_num = st.slider('Select educational number:', min_value=1, max_value=16, value=10)

    # Hours per week radio buttons
    hours_per_week = st.radio('Select hours per week:', ['parttime', 'fulltime'])


    # Create input data based on user selections
    input_data = {
        'age': age,
        'marital-status': marital_status,
        'occupation': occupation,
        'educational-num': educational_num,
        'hours-per-week': hours_per_week
    }

    # Predict function
    def predict_income(data):
        # Prepare input data for prediction
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return prediction[0]

    # Prediction button
    if st.button('Predict Income Class'):
        prediction = predict_income(input_data)
        income_class = '>=50K' if prediction == 1 else '<50K'
        st.header(f'The predicted income : {income_class}')

if __name__ == '__main__':
    main()


