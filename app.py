import streamlit as st
import pandas as pd
import joblib
import time

# настройка страницы
st.set_page_config(page_title="Предсказатель дохода", page_icon="💰")

# 1. загружаем модель
@st.cache_resource
def load_model():
    return joblib.load('model_income.pkl')
data = load_model()
model = data['model']
model_columns = data['columns']

# 2. интерфейс
st.title('Пробиваем стеклянный потолок в $50k?')
st.markdown("""
Это ML-модель, которая проанализирует ваши данные и предскажет, **превысит ли ваш годовой доход $50,000**.
""")

st.divider()

st.sidebar.header('Личный профиль')

age = st.sidebar.slider('Возраст', 17, 90, 30)
sex = st.sidebar.radio('Пол', ['Male', 'Female'], horizontal=True, format_func=lambda x: "Мужской" if x == 'Male' else "Женский")
race = st.sidebar.selectbox('Раса / Этнос', 
    ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
relationship = st.sidebar.selectbox('Роль в семье', 
    ['Husband', 'Wife', 'Own-child', 'Unmarried', 'Not-in-family', 'Other-relative'])
marital_status = st.sidebar.selectbox('Семейное положение', 
    ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])


st.subheader("Работа и образование")

col1, col2 = st.columns(2)

with col1:
    education = st.selectbox('Образование', 
        ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    
    workclass = st.selectbox('Тип занятости', 
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])

with col2:
    occupation = st.selectbox('Сфера деятельности', 
        ['Exec-managerial', 'Prof-specialty', 'Tech-support', 'Sales', 'Craft-repair', 'Other-service', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    
    hours_per_week = st.slider('Часов работы в неделю', 1, 99, 40)

education_num = st.slider('Общее количество лет обучения', 1, 16, 10)

# думаю спрятать, так как особо не применимо
with st.expander("Дополнительные доходы/убытки (инвестиции и т.д.)"):
    c1, c2 = st.columns(2)
    capital_gain = c1.number_input('Прирост капитала ($)', value=0, step=1000)
    capital_loss = c2.number_input('Потеря капитала ($)', value=0, step=1000)

# среднее значение по датасету берем
fnlwgt = 189154.5339154232 

# 3. Предсказание
st.markdown("###") 
if st.button('Предсказать доход', type='primary', use_container_width=True):
    

    # мсбор данных
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week}
    
    input_df = pd.DataFrame([input_data])

    input_df_encoded = pd.get_dummies(input_df, drop_first=False) 

    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
    # gредсказание
    prediction = model.predict(input_df_encoded)
    probability = model.predict_proba(input_df_encoded)[0][1]
    
    st.divider()

    col_res, col_metric = st.columns([2, 1])
    
    with col_res:
        if prediction[0] == 1:
            st.success('**Успех!** Модель считает, что ваши характеристики соответствуют **высокому уровню дохода (>50k)**')
            st.balloons()
        else:
            st.warning('Модель предсказывает доход **до $50k**.')
            st.caption("Не надо грустить. Глупые дата-сатанисты все равно все выдумывают")

    with col_metric:
        st.metric(label="Вероятность >50k", value=f"{probability:.1%}")
        st.progress(probability)