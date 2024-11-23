import streamlit as st
import pickle
import pandas as pd

# Cargar los pickles
with open('TrabajoBPA/Modelo.pickle', 'rb') as f:
    modelo = pickle.load(f)

with open('TrabajoBPA/MinMax.pickle', 'rb') as f:
    minmax_scaler = pickle.load(f)

with open('TrabajoBPA/oneHE.pickle', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('TrabajoBPA/ordEN.pickle', 'rb') as f:
    ordinal_encoder = pickle.load(f)

# Columnas asociadas a cada transformador
columnas_minmax = [
    'Edad', 'IMC', 'Consumo_Alcohol', 'Nivel_Actividad_Fisica',
    'Calidad_Dieta', 'Calidad_Sueno', 'Presion_Sistolica',
    'Presion_Diastolica', 'Colesterol_Total', 'Colesterol_LDL',
    'Colesterol_HDL', 'Trigliceridos_Colesterol',
    'Examen_Cognitivo_MMSE', 'Evaluacion_Funcional',
    'Actividades_Vida_Diaria'
]

columnas_onehot = [
    'Historial_Familiar_Alzheimer', 'Enfermedad_Cardiovascular',
    'Diabetes', 'Depresion', 'Lesion_Craneal', 'Hipertension',
    'Quejas_Memoria', 'Problemas_Comportamiento', 'Confusion',
    'Desorientacion', 'Cambios_Personalidad',
    'Dificultad_Completar_Tareas', 'Olvidos'
]

columnas_ordinal = ['Nivel_Educativo']
niveles_educativos = ['Ninguno', 'Escuela secundaria', 'Superior', 'Licenciatura']

# Interfaz de usuario en Streamlit
st.set_page_config(page_title="Predicción de Alzheimer", page_icon="🧠", layout="centered")
st.title("Predicción de Alzheimer 🧠")
st.markdown("""
    ¡Bienvenido a nuestra herramienta de predicción!  
    Ingrese los datos del paciente para obtener una predicción sobre la posibilidad de Alzheimer.  
    El modelo te dará una probabilidad sobre si el paciente tiene Alzheimer. 😊
""")

st.header("Información del Paciente")
genero = st.selectbox("Género:", ["Femenino", "Masculino"])
nivel_educativo = st.selectbox("Nivel Educativo:", niveles_educativos)
datos_minmax = {}
for col in columnas_minmax:
    datos_minmax[col] = st.number_input(f"{col}:", min_value=0.0, max_value=1000.0, step=0.1)
fuma = st.selectbox("¿Fuma?:", ["Si", "No"])

st.header("Historial de Padecimientos y Síntomas")
datos_onehot = {}
for col in columnas_onehot:
    datos_onehot[col] = st.selectbox(f"{col}:", ["No", "Si"])

# Botón para realizar la predicción
if st.button("Predecir", use_container_width=True):
    # Preparar datos para MinMaxScaler
    df_minmax = pd.DataFrame([datos_minmax])
    datos_minmax_transformados = minmax_scaler.transform(df_minmax)

    # Preparar datos para OneHotEncoder
    df_onehot = pd.DataFrame([datos_onehot])
    datos_onehot_transformados = onehot_encoder.transform(df_onehot)

    # Preparar datos para OrdinalEncoder
    df_ordinal = pd.DataFrame([[nivel_educativo]], columns=columnas_ordinal)
    datos_ordinal_transformados = ordinal_encoder.transform(df_ordinal)

    # Transformar Género y Fuma a dummy variables
    genero_femenino = 1 if genero == "Femenino" else 0
    genero_masculino = 1 if genero == "Masculino" else 0
    fumador_fuma = 1 if fuma == "Si" else 0
    fumador_no_fuma = 1 if fuma == "No" else 0

    # Crear DataFrame con las columnas dummy
    datos_dummies = pd.DataFrame({
        'Genero_Femenino': [genero_femenino],
        'Genero_Masculino': [genero_masculino],
        'Fumador_Fuma': [fumador_fuma],
        'Fumador_No fuma': [fumador_no_fuma]
    })

    # Combinar todas las transformaciones en un solo array
    datos_combinados = pd.concat(
        [
            datos_dummies,
            pd.DataFrame(datos_onehot_transformados, columns=onehot_encoder.get_feature_names_out(columnas_onehot)),
            pd.DataFrame(datos_ordinal_transformados, columns=columnas_ordinal),
            pd.DataFrame(datos_minmax_transformados, columns=columnas_minmax)  # Agregar las columnas dummies de Género y Fuma
        ],
        axis=1
    )
    
    # Realizar predicción
    prediccion = modelo.predict(datos_combinados)
    
    # Obtener las probabilidades de las clases
    probabilidades = modelo.predict_proba(datos_combinados)
    
    # Mostrar el resultado con la probabilidad
    if prediccion[0] == 1:
        st.markdown(f"### 🧠 El modelo predice que el paciente podría tener Alzheimer.")
        st.markdown(f"**Probabilidad:** {probabilidades[0][1]:.2f} (alta posibilidad de Alzheimer)")
        st.success("¡Cuidemos la salud cognitiva! Mantén un seguimiento adecuado.")
    else:
        st.markdown(f"### 😊 El modelo predice que el paciente no tiene Alzheimer.")
        st.markdown(f"**Probabilidad:** {probabilidades[0][0]:.2f} (baja probabilidad de Alzheimer)")
        st.success("¡Excelente! Continúa con un estilo de vida saludable.")
