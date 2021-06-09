import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
  page_title="Titanic app",
 layout="wide",
 
)
st.title('Survivrez vous au Titanic ?')

st.markdown("""
Envie de savoir si vous auriez survécu au Titanic ? Plutôt Rose ou Jack ?
* **Data source:** Kaggle : Titanic project
* **Auteur** : Vianney BONTE
* **Python libraries:** pandas, streamlit, pickle, sklearn
""")

st.sidebar.image(r'images/titanic.jpg', width=300)

sexe_lbl = st.sidebar.selectbox('Homme ou Femme ?', ["Homme", "Femme"])
richesse_lbl = st.sidebar.selectbox('Riche, moyen ou pauvre ?', ["Riche","Moyen", "Pauvre"])
age = st.sidebar.slider('Quel âge avez vous ?', 1, 110, 1, 1)
famille = st.sidebar.slider('Combien de menbre de votre famille à aider ?', 1, 6, 1, 1)
fare = st.sidebar.slider('Prix de votre ticket ?', 1, 250, 1, 1)

if sexe_lbl =="Homme":
    sexe = 0
else:
    sexe = 1

if richesse_lbl =="Riche":
    Pclass = 1
elif richesse_lbl =="Moyen":
    Pclass = 2
else :
    Pclass = 3


Pkl_Filename = r"modele_knn_80.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    pickled_model = pickle.load(file)


st.write("Le modèle utilisé :")
pickled_model

st.write("Ce modèle est fiable à 80% ")

data = {'pclass':[Pclass],'sex':[sexe],'age':[age], 'sibsp':[famille], 'fare':[fare]}
df = pd.DataFrame(data=data)

df = MinMaxScaler().fit_transform(df)

df = pd.DataFrame(df, columns = ['pclass', 'sex', 'age', 'sibsp', 'fare'])

df.dropna(axis=0, inplace=True)

Ypredict = pickled_model.predict(df)  


if Ypredict == 1:
    st.write("Le résultat :")
    col1, mid, col2 = st.beta_columns([1,1,200])
    with col1:
        st.image('images/gg.jpg', width=200)
    with col2:
        st.write('Bravo ! Vous survivez !')


elif Ypredict == 0:
    st.write("Le résultat :")
    col1, mid, col2 = st.beta_columns([1,1,200])
    with col1:
        st.image('images/rip.jpg', width=200)
    with col2:
        st.write('R.I.P.')





