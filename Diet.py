import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('Personalized_Diet_Recommendations.csv')


data['Chronic_Disease'] = data['Chronic_Disease'].fillna('No')


data = data.drop(columns=['Allergies', 'Food_Aversions'], errors='ignore')

# Label Encoding categorical features
le_gender = LabelEncoder()
le_chronic = LabelEncoder()
le_dietary = LabelEncoder()
le_cuisine = LabelEncoder()
le_meal = LabelEncoder()

data['Gender_enc'] = le_gender.fit_transform(data['Gender'])
data['Chronic_enc'] = le_chronic.fit_transform(data['Chronic_Disease'])
data['Dietary_enc'] = le_dietary.fit_transform(data['Dietary_Habits'])
data['Cuisine_enc'] = le_cuisine.fit_transform(data['Preferred_Cuisine'])
data['Meal_enc'] = le_meal.fit_transform(data['Recommended_Meal_Plan'])

# Features and target
features = [
    'Age', 'BMI', 'Daily_Steps', 'Exercise_Frequency', 'Sleep_Hours',
    'Gender_enc', 'Chronic_enc', 'Dietary_enc', 'Cuisine_enc', 'Caloric_Intake'
]
X = data[features]
y = data['Meal_enc']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=700,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🥗 Personalized Nutrition & Meal Plan Recommendation")
st.sidebar.header("Enter Your Health Details")

# Collect user input
age = st.sidebar.slider("Age", 10, 100, 30)
height = st.sidebar.slider("Height (cm)", 100, 220, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
bmi = round(weight / ((height / 100) ** 2), 2)
daily_steps = st.sidebar.slider("Daily Steps", 0, 20000, 7000)
exercise = st.sidebar.slider("Exercise Frequency (days/week)", 0, 7, 3)
sleep_hours = st.sidebar.slider("Sleep Hours", 0, 12, 7)
gender = st.sidebar.selectbox("Gender", data['Gender'].unique())
chronic = st.sidebar.selectbox("Chronic Disease", data['Chronic_Disease'].unique())
dietary = st.sidebar.selectbox("Dietary Habits", data['Dietary_Habits'].unique())
cuisine = st.sidebar.selectbox("Preferred Cuisine", data['Preferred_Cuisine'].unique())
calories = st.sidebar.number_input("Daily Caloric Intake", 1000, 5000, 2000)

# Encode input
input_data = [[
    age, bmi, daily_steps, exercise, sleep_hours,
    le_gender.transform([gender])[0],
    le_chronic.transform([chronic])[0],
    le_dietary.transform([dietary])[0],
    le_cuisine.transform([cuisine])[0],
    calories
]]


prediction = model.predict(input_data)[0]
recommended_meal = le_meal.inverse_transform([prediction])[0]


st.subheader("✅ Recommended Meal Plan:")
st.success(recommended_meal)

if st.checkbox("Show Full Dataset"):

    st.dataframe(data)
