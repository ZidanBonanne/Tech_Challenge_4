import streamlit as st


import pandas as pd


st.write("""
    ## Tech Challenge #04 - Grupo 9 
    ### Modelo Preditivo / Petróleo Brent
""")

st.write("""
    ## Os integrantes do grupo são:
    Aline Lima Moreira Oliveira\n
    Felipe Silveira Stopiglia\n
    Flávio Laerton Seixas Castro\n
    Guilherme de Lima Cerqueira\n
    Zidan Bonanne Franco Rocha\n
""")
st.title('O que é o Tech Challenge?')
st.markdown(
    """
    Tech Challenge é o projeto que englobará os conhecimentos obtidos em todas as disciplinas da fase;
    O objetivo do desafio é desenvolver um dashboard interativo com um modelo de machine learning para prever o preço diário do petróleo utilizando series temporais. O dashboard deve fornecer insights que possam colaborar com as oscilações que os preços tiveram, visando esclarecer o mercado analisado.
"""
)
st.image("https://kairamcabral.com.br/wp-content/uploads/2020/02/COMO-LIDAR-COM-OS-DESAFIOS-NA-VIDA-1600x800.jpg")  
# st.info(f"""
#     Com objetivo de predizer o valor do Petróleo Brent, mostramos nesse trabalho
#     todo o processo para criação do nosso modelo, e algumas análises do histórico do mesmo.
    
#     Os dados aqui utilizados foram baixados do site [IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view) 
#     e contemplam o período de {min(df_petroleo_recorte_2023.ds).date()} até {max(df_petroleo_recorte_2023.ds).date()}.
# """)
        

# tab_grafico_historico, tab_seasonal, tab_adf, tab_acf = st.tabs(['Gráfico Histórico', 'Decompondo sazonalidade', 'Teste ADFuller', 'Autocorrelação - ACF/PACF'])


# with tab_seasonal:
#     subtab_dec_basica, subtab_180, subtab_365 = st.tabs(['Decomposição Básica', '180 dias','365 dias'])

#     with subtab_dec_basica:
#         st.markdown("""
#             Utilizando a função `seasonal_decompose` sem parâmetros, não foi identificado nenhum padrão sazonal.
#         """)
#         st.plotly_chart(
#             generate_graphs._seasonal_decompose(get_data._series_for_seasonal()),
#             use_container_width=True,
#         )

#     with subtab_180:
#         st.markdown("""
#             Ao tratarmos o periodo como 180, porém, notamos um comportamento sazonal mais evidente:
#         """)
#         st.plotly_chart(
#             generate_graphs._seasonal_decompose(get_data._series_for_seasonal(), 180),
#             use_container_width=True,
#         )

#     with subtab_365:
#         st.markdown("""
#             Ainda, ao tratarmos o periodo como 365, porém, notamos um comportamento sazonal mais evidente:
#         """)
#         st.plotly_chart(
#             generate_graphs._seasonal_decompose(get_data._series_for_seasonal(), 365),
#             use_container_width=True,
#         )
        
#         st.markdown("""
#             Seria um sinal de que temos um comportamento cíclico anual, e uma tendência bem mais definida?
#             Nota-se que o gráfico de tendência está muito mais constante e conciso.
#         """)

# with tab_adf:

#     st.markdown(f"""
#     Este é um teste estatístico utilizado na análise de séries temporais para determinar 
#     se uma determinada série temporal é estacionária ou não.
#     """)
    
#     grafico_adf, series_adf = generate_graphs._adf(df_petroleo)
#     res_adf = get_data._adfuller(series_adf)
#     st.plotly_chart(
#         grafico_adf,
#         use_container_width=True,
#     )

#     st.markdown(f"""
#         Aplicando a função `adfuller` ao dados sem nenhum tratamento, verificamos que a série não é estacionária
#         ```
#         Teste estatístico: {res_adf[0]}
#         p-value: {res_adf[1]}
#         Valores críticos: {res_adf[4]}
#         ```
#     """)

#     st.divider()

#     grafico_adf_diff, series_adf_diff = generate_graphs._adf_diff(df_petroleo)
#     res_adf_diff = get_data._adfuller(series_adf_diff)
#     st.plotly_chart(
#         grafico_adf_diff,
#         use_container_width=True,                                 
#     )

#     st.markdown(f"""
#         Normalizando os dados com a diferenciação conseguimos transformar a série em estacionária.
#         ```
#         Teste estatístico: {res_adf_diff[0]}
#         p-value: {res_adf_diff[1]}
#         Valores críticos: {res_adf_diff[4]}
#         ```
#     """)

# with tab_acf:
#     st.pyplot(
#         plot_acf(df_petroleo['Preco'].values, lags=1460),
#         use_container_width=True,
#     )
#     st.markdown("""
#     Para a autocorrelação, vemos um comportamento significativo até aproximadamente 2 anos, ficando cada vez menos significativo a partir desse ponto... 
#     Será que tinhamos um comportamento diferente após esse tempo? 
#     O que disparou essa diferença no comportamento, a ponto de não ser estatisticamente significante a relação entre os valores atuais e de 2 anos anteriores?
#     """)

#     st.divider()

#     st.pyplot(
#         plot_pacf(df_petroleo['Preco'].values, lags=30),
#         use_container_width=True,
#     )
#     st.markdown("""
#     Para a autocorrelação Parcial, vemos um comportamento de auto regressão com um termo de ordem 5 - dada a proximidade do limiar da insignificância no termo de ordem 6. Poderíamos utilizar, portanto, modelos auto-regressivos e com presença de médias móveis, como o ARMA/ARIMA, ou modelos especializados em séries temporais, especificamente, com resultados satisfatórios.

#     Iniciaremos, portanto, uma análise de componentes históricos que possam influenciar no valor do petróleo, passando para uma modelagem de séries temporais, e alguns testes de importância de variáveis para confirmar as hipóteses e análises demonstradas, em sequência. 
#     """)

