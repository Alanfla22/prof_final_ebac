
import timeit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import pickle
import cloudpickle
import joblib
from PIL import Image

custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)

model = pickle.load(open('./modelo/model_final.pkl', 'rb'))



# Função para ler os dados
@st.cache_data
def load_data(file_data):
  
  df = pd.read_csv(file_data)
  df.drop_duplicates(inplace=True)

  return df  

# função para converter dataframe em arquivo csv
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False)

def remove_columns(X):

  X = X.drop(['data_ref', 'index'], axis=1)

  return X   

def remove_outliers(X):

  # Calcular os quartis Q1 e Q3
  Q1 = X.quantile(0.25)
  Q3 = X.quantile(0.75)

  # Calcular o IQR
  IQR = Q3 - Q1

  # Definir os limites inferior e superior
  limite_inferior = Q1 - 1.5 * IQR
  limite_superior = Q3 + 1.5 * IQR

  # Remover outliers
  X_new = X[(X >= limite_inferior) & (X <= limite_superior)]

  return X_new  

pipe = joblib.load('./pipeline/pipe.joblib')

# função de predição
@st.cache_resource()
def predict(df_input):

  df_x = pipe.transform(df_input)

  prob_mau = model.predict_proba(df_x)[:,1]
  
  df_input['mau'] = prob_mau > 0.50
  
  df_input['risco'] = 'BAIXO'
  df_input.loc[prob_mau >= 0.30, 'risco'] = 'MÉDIO'
  df_input.loc[prob_mau > 0.50, 'risco'] = 'ALTO'
  
  return df_input

  

def main():
    st.set_page_config(page_title = 'Análise de Risco de Crédito', \
        page_icon = './images/telmarketing_icon.png',
        layout ='wide',
        initial_sidebar_state='expanded')
    st.title('ANÁLISE DO RISCO DE CRÉDITO')
    st.subheader('', divider='rainbow')

    image = Image.open('./images/imagem risco.png')
    st.sidebar.image(image)

    st.sidebar.write("## Upload do arquivo")
    data_file_1 = st.sidebar.file_uploader("", label_visibility="hidden",
                                            type=['csv','xlsx'])

    data_processado = ''
    
    
    if (data_file_1 is not None):
        
        bank_raw = load_data(data_file_1) 

        st.sidebar.write("## Classificar o risco")
        with st.sidebar.form(key='my_form'):

            
          submitted = st.form_submit_button(label='Aplicar')         

          if submitted:
            
            data_processado = predict(bank_raw)  
      

    try:

      quant_registro = bank_raw.shape[0]

      frase = f'Quantidade de Registros*: {quant_registro}'

      st.subheader(frase)
      st.write('* Exceto duplicidades')

      col1, col2 = st.columns([2, 3])

      with col1:

        st.header('PERFIL DOS CLIENTES')
        lista_cat = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']
        tabs = st.tabs(lista_cat)

        for i in range(len(lista_cat)):
                           
          with tabs[i]:
              
            fig = px.pie(bank_raw, names=lista_cat[i], color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True) 

      with col2:

        st.header('')  
        lista_num = ['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda']
        tabs = st.tabs(lista_num)

        for i in range(len(lista_num)):
                           
          with tabs[i]:
              
            fig = px.histogram(bank_raw, x=lista_num[i], color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True) 


      
      try:

        data_processado['risco'] = data_processado['risco']

        col1, col2 = st.columns([2, 3])

        with col1:

          st.markdown('#### Classificação dos Clientes')

          tab1, tab2, tab3 = st.tabs(["Baixo", "Médio", "Alto"])

          with tab1:

            st.dataframe(data_processado[data_processado['risco'] == 'BAIXO'])
          
          with tab2:

            st.dataframe(data_processado[data_processado['risco'] == 'MÉDIO'])

          with tab3:

            st.dataframe(data_processado[data_processado['risco'] == 'ALTO'])  
                  
        with col2:

          st.markdown('#### Composição do Risco')
            
          tab1, tab2 = st.tabs(["Quantidade", "Proporção"])

          with tab1:

            fig_1 = px.histogram(data_processado, x='risco', color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_1, use_container_width=True)

          with tab2:

            fig_2 = px.pie(data_processado, names='risco', color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_2, use_container_width=True)   
        
        st.markdown('### Baixe aqui os resultados')
        data_processado_csv = convert_df(data_processado)      
        st.download_button(label='📥 Download',
                          data=data_processado_csv,
                          file_name= 'data_processado.csv')

      except:

        st.write('## Faça a classificação de risco')                    

    except:

      st.write('## Carregue os dados')
     

if __name__ == '__main__':
	main()