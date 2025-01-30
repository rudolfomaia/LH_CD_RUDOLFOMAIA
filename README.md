# Projeto LIGHTHOUSE Indicium
Repositório com os arquivos do desafio DS do programa LIGHTHOUSE da empresa Indicium

## Desafio

Você foi alocado(a) em um time da Indicium que está trabalhando atualmente junto a um cliente no processo de criação de uma plataforma de aluguéis
temporários na cidade de Nova York. Para o desenvolvimento de sua estratégia de precificação, pediu para que a Indicium fizesse uma análise exploratória
dos dados de seu maior concorrente, assim como um teste de validação de um modelo preditivo.

## Objetivo

Seu objetivo é desenvolver um modelo de previsão de preços a partir do dataset oferecido, e avaliar tal modelo utilizando as métricas de avaliação
que mais fazem sentido para o problema.

## Índice

1. [configuração](#configuracao)
2. [Aquisição de Dados](##aquisicao-de-dado)
3. [Limpeza para Análise]("#limpeza-para-analise")
4. [Análise Exploratória de Dados (EDA)](#analise-exploratoria-de-dados-eda)
5. [Pré-processamento e treinamento de ML](#pre-processamento-treinamento)

## 1. Configuração
<a id="#configuracao"></a>

Bibliotecas Utilizadas

Para este projeto, utilizaremos as seguintes bibliotecas:

- **catboost:** Algoritmo de boosting, variáveis categóricas.
- **collections:** Estruturas de dados.
- **lightgbm:**aprendizado de máquina.
- **matplotlib:** Criação de gráficos.
- **numpy:** Operações matemáticas.
- **pandas:** Gerenciar os dados.
- **pickle:** Serialização e desserialização de objetos.
- **plotly:** Ferramentas visualização de dados.
- **re:** Manipulação de expressões regulares.
- **scikit-learn:** 
  - **sklearn.cluster:** Algoritmos de agrupamento.
  - **sklearn.ensemble:** Algoritmos de ensemble.
  - **sklearn.linear_model:** Modelos lineares.
  - **sklearn.metrics:** Medidas de performance dos modelos.
  - **sklearn.model_selection:** Métodos de validação cruzada e busca de hiperparâmetros.
  - **sklearn.neighbors:** Algoritmos de vizinhança.
  - **sklearn.preprocessing:** Técnicas de pré-processamento de dados.
- **scipy:** 
  - **scipy.stats:** Funções estatísticas.
- **seaborn:** Gráficos estatísticos baseados no matplotlib.
- **statsmodels:** Estimativas estatísticas, testes e modelos para dados.
- **wordcloud:** Geração de nuvens de palavras.
- **xgboost:** Árvores de decisão reforçadas.<br>

Caso não possua alguma das bibliotecas instalada em sua maquina, basta abrir um novo terminal e executar a instalação através do comando "pip install nome-biblioteca".
Exemplo: pip install pandas

## 2. Aquisição de Dados
<a id="#aquisicao-de-dados"></a>

<p>
<b>Dados do suposto concorrente do cliente final, fornecidos pela Indicium (dataset).</b> <br>
<ul>
    <li>Fonte de dados: <a href="https://drive.google.com/drive/folders/1osZizjZ-hd0SMD5J6-vUZvB8bMJI3zVV" target="_blank">https://drive.google.com/drive/folders/1osZizjZ-hd0SMD5J6-vUZvB8bMJI3zVV</a></li>
    <li>tipo de dados: csv</li>
    <li>Licença: A licença para este conjunto de dados não é especificada</li>
</ul>
<b>Localização das escolas públicas de NY.</b> <br>
<ul>
    <li>Fonte de dados: <a href="https://data.cityofnewyork.us/Education/NYC-DOE-Public-School-Location-Information/3bkj-34v2/about_data" target="_blank">https://data.cityofnewyork.us/Education/NYC-DOE-Public-School-Location-Information/3bkj-34v2/about_data</a></li>
    <li>tipo de dados: csv</li>
    <li>Licença: A licença para este conjunto de dados não é especificada</li>
</ul>
<b>Localização centros públicos computacionais.</b><br>
(centros públicos com computador e internet) <br>
<ul>
    <li>Fonte de dados: <a href="https://data.cityofnewyork.us/Social-Services/Citywide-Public-Computer-Centers/sejx-2gn3/about_data" target="_blank">https://data.cityofnewyork.us/Social-Services/Citywide-Public-Computer-Centers/sejx-2gn3/about_data</a></li>
    <li>tipo de dados: csv</li>
    <li>Licença: A licença para este conjunto de dados não é especificada</li>
</ul>
<b>Resumo de venda de imóveis por bairro</b><br>
<ul>
    <li>Fonte de dados: <a href="https://data.cityofnewyork.us/City-Government/DOF-Summary-of-Neighborhood-Sales-by-Neighborhood-/5ebm-myj7/about_data" target="_blank">https://data.cityofnewyork.us/City-Government/DOF-Summary-of-Neighborhood-Sales-by-Neighborhood-/5ebm-myj7/about_data</a></li>
    <li>tipo de dados: csv</li>
    <li>Licença: A licença para este conjunto de dados não é especificada</li>
</ul>
</p>

## 3. Limpeza para Análise
<a id="#limpeza-para-analise"></a>

<p>
Análise das tabelas de dados, identificação de células vazias, colunas irrelevantes para o projeto, etc. <br>
</p>

<b>Tratamento da base Dataset:</b>

Algumas colunas são irrelevantes ao preço do aluguel, pois existem apenas para um controle interno do servidor de dados. 

<ul>
    <li>Exclusão das colunas: id, host_id, host_name, ultima_review.</li>
    <li>Exclusão de linhas: Remover as linhas vazias.</li>
</ul>

<b>Observações</b>
<ul>
    <li>A coluna "nome" será mantida a pedido do cliente que deseja saber: "Existe algum padrão no texto do nome do local para lugares de mais alto valor?"</li>
    <li>As colunas: disponibilidade_365, calculado_host_listings_count, numero_de_reviews, serão mantidas por serem do tipo Inteiros e não terem linhas vazias, podem auxiliar na fase de aprendizagem e DEA, e poderão nos mostrar algo nos graficos.</li>
    <li>A coluna "reviews_por_mes" foi realizada uma analise separada, pois tem 1052 linhas vazias, que após uma analize das 50 primeiras linhas vazias, mostrou que excluir essas linhas para manter essa coluna é inviavél. Para manter essa coluna temos que utilizar de imputação de valores: Com Mediana(0.72) ou Clusters com KNN, porém após avaliar os outliers, teria que ser aplicado limittação(capping) ou transformação para tratamento. Em resumo 1/4 dos dados dessa coluna perderia a confiabilidade, ela não é uma coluna que valha tamanho risco na credibilidade futura, decisão final exclusão da coluna.</li>
    <li> Após utilizar a coluna prices em alguns graficos, identificado Outliers, devido a inscosistencia nas analises futuras e a quantidade ser abaixo de 10%, optei por remover as linhas e não transformalas.
</ul>

<b>Tratamento da base escolas de NY e centros computacionais:</b>

A base com a localização das escolas servirá de análise para identificar se tem ou não relevância escolas pertos do imóvel no preço do aluguel.

<ul>
    <li>Exclusão de colunas: remoção de todas as colunas exceto: Longitude, Latitude.</li>
    <li>Exclusão de linhas: Remover as linhas que não tem a localização.</li>
</ul>


<b>Tratamento da base resumo de venda de imóveis por bairro:</b>

A base com o resmudo de venda de imóveis por bairro, será utilizado para analisar e cruzar a informação do valor de venda em média e o valor médio de aluguel daquela bairro.

<ul>
    <li>Exclusão de colunas: remoção de todas as colunas exceto: BOROUGH, NEIGHBORHOOD, TYPE OF HOME, AVERAGE SALE PRICE, YEAR.</li>
    <li>Exclusão de linhas: Mantive apenas as linhas do ano de 2022.
</ul>

## 4. Análise Exploratória de Dados (EDA)
<a id="analise-exploratoria-de-dados-eda"></a>

<p>
Para a análise exploratoria de dados(EDA), realizei uma análise detalhada das variáveis do dataset para entender melhor a distribuição, correlação e possíveis impactos no preço de aluguéis. A seguir estão as principais etapas durante a análise.

**Análise Univariada**

Para iniciar, fiz uma análise univariada da variável price para entender sua distribuição. Utilizei histogramas e gráficos de densidade para visualizar a distribuição dos preços.

**Análise Bivariada**

Explorei as relações entre preço e outras variáveis categóricas e numéricas. Algumas análises incluíram:

- **Distribuição das variáveis categóricas**: 
    - Plotei histogramas para variáveis como room_type, bairro_group e bairro.
- **Relação entre latitude e longitude com o preço**:
    - Utilizei scatter plots coloridos por preço para visualizar a distribuição geográfica dos aluguéis.
- **Boxplots**:
    - Analisei a relação de preço com room_type, bairro_group e bairro.
- **Correlação de Pearson**:
    - Calculei e plotei a correlação de Pearson entre distancia_escola_mais_proxima e preci, além de outras variáveis numéricas.

**Análise Multivariada**

Realizei análises multivariadas para entender as interações entre múltiplas variáveis. Utilizei gráficos de pares e matrizes de correlação para identificar possíveis relações:

- **Pairplots**:
    - Visualizei a interação entre diversas variáveis numéricas.
- **Heatmaps de correlação**:
    - Gerei mapas de calor para analisar a correlação entre as variáveis numéricas.

**Análise de Correlação**

Utilizei uma matriz de correlação para calcular as correlações de Pearson entre variáveis importantes, permitindo identificar relações significativas que possam impactar o preço dos aluguéis.

**Normalização de Dados**

Para garantir a consistência das análises, realizei a normalização dos dados:

- **StandardScaler**: Utilizado para colunas com distribuições próximas do normal, como numero_de_reviews, disponibilidade_365 e calculado_host_listings_count.
- **MinMaxScaler**: Aplicado para colunas com distribuições enviesadas ou valores extremos, como minimo_noites, distancia_escola_mais_proxima e distancia_pcc_mais_proximo.

**Análise da Variável Nome**

Atendendo a uma solicitação específica, analisei detalhadamente a variável nome para localizar padrões:

- **Contagem de palavras**:
    - Excluí palavras irrelevantes e analisei a frequência das palavras mais comuns nos nomes das propriedades.
- **Comprimento do Nome**:
    - Analisei a relação entre o comprimento dos nomes e o preço dos aluguéis.
</p>

## 5. Pré-processamento e treinamento de ML
<a id="pre-processamento-treinamento"></a>

<p>
Apresento as técnicas de modelagem e validação utilizadas para prever o preço dos aluguéis com base nas variáveis do dataset. As seguintes etapas foram realizadas:

**Regressão Linear Múltipla**

Realizei a modelagem usando Regressão Linear Múltipla para quantificar a relação entre o preço e as variáveis independentes.

- **Variáveis Independentes**: numero_de_reviews, disponibilidade_365, calculado_host_listings_count, minimo_noites, distancia_escola_mais_proxima, distancia_pcc_mais_proximo;
- **Variável Dependente**: price;
- **Resultados**: 
  - **R²**: 3.9%
  - **Erro Médio Absoluto**: N/A (não calculado)

K-Means Clustering
Apliquei K-Means para agrupar imóveis com base em características semelhantes.
- **Número de Clusters**: 3 (determinado pelo método do cotovelo)
- **Visualização**: Clusters plotados usando PCA para redução de dimensionalidade

**Nota** Preferi remover a distância dos centros e escolas, pois não impactava no preço e para agilizar o processamento dos modelos.

**Random Forest Regressor**
Utilizei o modelo Random Forest Regressor para capturar interações não lineares.
- **Erro Médio Absoluto**: 32.1665
- **R²**: 55.89%

**Validação Cruzada com Random Forest**
Adicionei validação cruzada ao modelo Random Forest para evitar overfitting.
- **Erro Médio Absoluto (Validação Cruzada)**: 33.5805
- **R² (Validação Cruzada)**: 53.28%

**Gradient Boosting Regressor (XGBoost)**
Testei o modelo Gradient Boosting Regressor para capturar relações mais complexas.
- **Erro Médio Absoluto**: 31.8374
- **R²**: 57.05%
- **Erro Médio Absoluto (Validação Cruzada)**: 33.1092
- **R² (Validação Cruzada)**: 54.75%

**Light Gradient Boosting Machine (LightGBM)**
Experimentei o modelo LightGBM, que apresentou os melhores resultados.
- **Erro Médio Absoluto**: 31.8446
- **R²**: 57.12%
- **Erro Médio Absoluto (Validação Cruzada)**: 32.9140
- **R² (Validação Cruzada)**: 55.29%

**LightGBM com RandomizedSearchCV**
Realizei a otimização dos hiperparâmetros usando RandomizedSearchCV devido à complexidade computacional.
- Melhores Parâmetros: {'num_leaves': 50, 'n_estimators': 500, 'min_child_samples': 40, 'max_depth': 30, 'learning_rate': 0.05}
- **Erro Médio Absoluto**: 31.4509
- **R²**: 57.71%

**Uso dos Arquivos PKL**
Para facilitar o uso do modelo otimizado e assegurar a reprodução dos resultados, três arquivos .pkl foram gerados:
1. **Modelo Otimizado (modelo_otimizado.pkl)**: Contém o modelo treinado com os melhores parâmetros.
2. **Normalizador (normalizador.pkl)**: Utilizado para normalizar os dados numéricos de entrada.
3. **Codificador (codificador.pkl)**: Responsável por codificar as variáveis categóricas.

**Como Utilizar o Modelo**
**Carregar os Arquivos PKL no inicio do cod de consulta**:
   
   with open('modelo_otimizado.pkl', 'rb') as arquivo_modelo:
       modelo_otimizado = pickle.load(arquivo_modelo)
   with open('normalizador.pkl', 'rb') as arquivo_normalizador:
       normalizador = pickle.load(arquivo_normalizador)
   with open('codificador.pkl', 'rb') as arquivo_codificador:
       codificador = pickle.load(arquivo_codificador)
</p>
