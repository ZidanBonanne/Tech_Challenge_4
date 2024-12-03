import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot, plot_components
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")


st.set_page_config(page_title="Modelo Prophet", page_icon="📈")

st.markdown("## Análises Realizadas para Construção do Modelo de Machine Learning")
st.title('Dados utilizados para o estudo')
df_petroleo = pd.read_csv('df_oil_tranformation.csv')
st.write(df_petroleo)
st.title('1.Tratamento dos dados para a Análise Exploratória')

st.markdown("##### Realizou-se a troca dos nomes das variáveis de price para y e date_price para ds para melhor utilização futura, além da conversão do campo ds para o tipo data.")



df_petroleo= df_petroleo.rename({'price':'y','date_price':'ds'},axis=1)

df_petroleo['ds'] = pd.to_datetime(df_petroleo['ds'],format='%Y-%m-%d')

st.write(df_petroleo)


st.title('2. Análise Exploratória dos Dados')
fig = plt.figure(figsize=(14, 7))
# ax.line_char

plt.plot(df_petroleo['ds'], df_petroleo["y"], label="Preço Petróleo")
plt.title("Análise da Série Temporal (DF Completo)")
plt.xlabel("Data")
plt.ylabel("Valores")
plt.legend("p")
st.pyplot(fig)


st.markdown("##### Decomposição da Série Temporal.")
df_petroleo.set_index('ds', inplace=True)
df_petroleo.sort_index(inplace=True)
decomposicao = seasonal_decompose(df_petroleo['y'], model='additive', extrapolate_trend='freq', period=180)

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize = (15,10))

decomposicao.observed.plot(ax=ax1)
decomposicao.trend.plot(ax=ax2)
decomposicao.seasonal.plot(ax=ax3)
decomposicao.resid.plot(ax=ax4)
plt.tight_layout()
st.pyplot(fig)

st.markdown("##### Estatísticas Descritivas.")
st.write(df_petroleo.describe())

st.title('3. Preparação dos Dados e Modelagem')
st.markdown("##### Inicialmente, foi importado um conjunto de dados com um histórico extenso; no entanto, para este projeto, utilizaremos apenas os dados a partir de 2023.")
df_petroleo_recorte_2023 = df_petroleo.loc[(df_petroleo.index >= '2023-01-01')]

df_petroleo_recorte_2023.plot(y='y',figsize=(12,6))
plt.xlabel('data')
plt.ylabel('valores em pontos')
plt.title('Visualização de pontos petroleo')
plt.legend('p')
plt.tight_layout()
st.pyplot(fig)

st.title('4. Aplicando o modelo Prophet')
decomposicao_2 = seasonal_decompose(df_petroleo_recorte_2023['y'], model='additive', extrapolate_trend='freq', period=180)

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize = (15,10))

decomposicao_2.observed.plot(ax=ax1)
decomposicao_2.trend.plot(ax=ax2)
decomposicao_2.seasonal.plot(ax=ax3)
decomposicao_2.resid.plot(ax=ax4)

plt.tight_layout()
st.pyplot(fig)
st.title('4.1. Plotando o gráfico para visualizar a perfomance do modelo.')
## Normalizando para aplicar o Prophet
df_petroleo_recorte_2023 = df_petroleo_recorte_2023.reset_index()

# importando o modelo e realizando o treinamento
model = Prophet(daily_seasonality=True,yearly_seasonality=True)
# inserido feriados no modelo, porém a base não tem feriados
model.add_country_holidays(country_name='BR')
model.fit(df_petroleo_recorte_2023)

# O forecast de 30 dias com o fechamento do índice petroleo
futuro = model.make_future_dataframe(30,freq='D')
# Realizando a predição
forecast = model.predict(futuro)

# Plotando as previsões
# plot(model, forecast)
fig1 = model.plot(forecast)
plt.axvline(x=df_petroleo_recorte_2023['ds'].iloc[-1], color='gray',linestyle='--')
plt.xlabel('Data')
plt.ylabel('Valores Previstos')
plt.title('Previsão de Série Temporal com Prophet')
plt.legend(['Pontos petroleo','Forecast','Intervalo Incerto','Começo Previsão'])
plt.tight_layout()
st.pyplot(fig)
st.markdown("##### Nesta outra visualização, podemos observar os movimentos de alta e baixa do preço, bem como as mudanças ao longo do tempo..")

fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)

plt.axvline(x=df_petroleo_recorte_2023['ds'][0], color='gray',linestyle='--')
plt.xlabel('Data')
plt.ylabel('Pontos petroleo')
plt.title('Forecast Realizado com Prophet petroleo')
plt.legend(['Pontos petroleo','Forecast','Intervalo Incerto','Começo Previsão','Mudança direção dos pontos'])
st.pyplot(fig)

st.title('5. Resultados de Perfomance')


st.markdown("##### validação cruzada do modelo para verificar suas principais métricas sendo o média do erro absoluto [mean absolute error], média do erro quadrático [mean squared error], média do erro percentual absoluto [mean absolute percentage error] e a raiz do erro quadrático médio [root mean squared error]. Modelo atual tem 476 datas, utilizaremos para o initial que seria o treinar do modelo 476 dias. Para o period utilizamos 7 dias que seriam o intervalo de datas utilizados para a cross validação e deve ser metade do horizon que definimos como 15 dias.")

df_cross_validation = cross_validation(model, initial='476 days', period='7 days', horizon='15 days')

# calcular as métricas do modelo
print("-------- Cálculo das Métricas --------")
df_metricas = performance_metrics(df_cross_validation)

# Calculate MAE, MSE, MAPE, and RMSE
mae = mean_absolute_error(df_cross_validation['y'], df_cross_validation['yhat'])
mse = mean_squared_error(df_cross_validation['y'], df_cross_validation['yhat'])
mape = mean_absolute_percentage_error(df_cross_validation['y'],df_cross_validation['yhat'])
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae:.0f}')
print(f'Mean Squared Error: {mse:.0f}')
print(f'Mean Absolute Percentage Error: {mape:.3f}')
print(f'Root Mean Squared Error: {rmse:.0f}')
print("-------- Fim da Validação --------")

st.title('6. Conclusão')

st.markdown("""##### Métricas do Modelo:""")
st.markdown("""Erro Absoluto Médio (MAE): 560\n
            Erro Quadrático Médio (MSE): 497154\n
            Erro Percentual Absoluto Médio (MAPE): 0.072\n
            Raiz do Erro Quadrático Médio (RMSE): 705\n
            A interpretação das métricas indica que, em média, o modelo apresenta um erro de 560 pontos. Esse resultado sugere que o modelo possui uma taxa de erro relativamente baixa e demonstra um bom desempenho na previsão dos dados.""")
 
