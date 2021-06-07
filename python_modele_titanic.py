import pandas as pd
import pickle
import streamlit as st

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


st.sidebar.image('images/titanic.jpg', width=300)

sexe_lbl = st.sidebar.selectbox('Homme ou Femme ?', ["Homme", "Femme"])
richesse_lbl = st.sidebar.selectbox('Riche, moyen ou pauvre ?', ["Riche","Moyen", "Pauvre"])
age = st.sidebar.slider('Quel âge avez vous ?', 1, 110, 1, 1)

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


Pkl_Filename = r"model_titanic.pkl"  


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    pickled_model = pickle.load(file)


st.write("Le modèle utilisé :")
pickled_model

st.write("Ce modèle est fiable à 78% ")


data = {'Pclass':[Pclass],'Sex':[sexe],'Age':[age]}
df = pd.DataFrame(data=data)

df.dropna(axis=0, inplace=True)
df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

Ypredict = pickled_model.predict(df)  


        
if Ypredict == 1:
    st.write("Le résultat :")
    col1, mid, col2 = st.beta_columns([1,1,100])
    with col1:
        st.image('images/gg.jpg', width=200)
    with col2:
        st.write('Bravo ! Vous survivez !')


else:
    st.write("Le résultat :")
    col1, mid, col2 = st.beta_columns([1,1,100])
    with col1:
        st.image('images/rip.jpg', width=200)
    with col2:
        st.write('R.I.P.')





