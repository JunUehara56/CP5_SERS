# CP5 - SERS | Machine Learning com Dados de Energia

Projeto desenvolvido para a disciplina de **Soluções em Energias Renováveis e Sustentáveis (SERS)**, do curso de **Ciência da Computação da FIAP**.

O projeto utiliza dados obtidos por meio de **duas APIs públicas** para desenvolver duas tarefas de Machine Learning utilizando Python:

1. classificação da fonte de geração de energia renovável;
2. regressão da radiação solar.

Em cada tarefa são utilizados três algoritmos diferentes, que são treinados, avaliados e comparados por meio de métricas adequadas para cada problema.

---

## Integrantes

- **Tommaso da C. Nagliatti** — RM 572147
- **Arthur Maziviero Faria** — RM 573928
- **Jun Uehara** — RM 570537
- **Felipe de Souza Gallo** — RM 569680
- **Matheus Martins Lacerda** — RM 570843
- **Roberson Reguero Luiz Junior** — RM 573031

---

## Estrutura do repositório

```text
CP5_SERS/
│
├── Aula_APIs_Energia_Renovavel_ML.ipynb
├── README.md
├── aneel_classificacao_orange.csv
└── meteo_regressao_orange.csv
```

---

# 1. Classificação da Fonte Renovável

## Fonte dos dados

Os dados utilizados nesta etapa são provenientes do **[SIGA — Sistema de Informações de Geração da ANEEL](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel)**.

Cada registro do conjunto de dados representa um empreendimento de geração de energia no Brasil.

As fontes foram agrupadas em três classes:

- **Solar** — empreendimentos UFV;
- **Eólica** — empreendimentos EOL;
- **Hidráulica** — empreendimentos UHE, PCH e CGH.

Os dados representam informações cadastrais dos empreendimentos e não correspondem à quantidade de energia efetivamente gerada.

---

## Estrutura dos dados

A partir dos dados obtidos pela API, foram selecionados e preparados os seguintes atributos para a tarefa de classificação:

| Atributo no CSV | Origem na API | Descrição | Papel |
|---|---|---|---|
| `potencia_kw` | `MdaPotenciaOutorgadaKw` | Potência outorgada em quilowatts; não representa energia produzida | Entrada |
| `latitude` | `NumCoordNEmpreendimento` | Latitude aproximada, em graus decimais | Entrada |
| `longitude` | `NumCoordEEmpreendimento` | Longitude aproximada, em graus decimais | Entrada |
| `fonte` | `SigTipoGeracao` | Categoria da fonte, agrupada em três classes | Alvo |

---

## Variáveis utilizadas

Para o treinamento dos modelos, `X` foi definido pelas três variáveis de entrada `potencia_kw`, `latitude` e `longitude`, enquanto `y` corresponde à coluna `fonte`, que representa a classe a ser prevista.

Não foram utilizadas como entradas variáveis que identificassem diretamente a fonte de geração, como siglas, nomes, códigos ou descrições.

---

## Análise inicial dos dados

Antes do treinamento dos modelos foi realizada uma análise exploratória do conjunto de dados.

Foram verificadas:

- quantidade de registros;
- tipos das variáveis;
- presença de valores ausentes;
- estatísticas descritivas;
- distribuição das classes.

O conjunto utilizado possui **3.876 registros**, distribuídos entre:

- **Hidráulica:** 1.476 registros;
- **Solar:** 1.200 registros;
- **Eólica:** 1.200 registros.

---

## Preparação dos dados

Os dados foram divididos em conjuntos de treino e teste utilizando:

- **80% dos registros para treinamento**;
- **20% dos registros para teste**;
- divisão estratificada pela variável `fonte`;
- `random_state=42` para garantir reprodutibilidade.

A padronização das variáveis foi realizada utilizando `StandardScaler`.

O scaler foi ajustado exclusivamente com os dados de treinamento e posteriormente aplicado aos dados de teste, evitando que informações do conjunto de teste fossem utilizadas durante o treinamento.

---

## Modelos de classificação

Foram utilizados três algoritmos diferentes:

### Logistic Regression

Modelo linear utilizado como referência para verificar a capacidade de separação das classes a partir das três variáveis de entrada.

### Gaussian Naive Bayes

Modelo probabilístico utilizado por trabalhar com variáveis numéricas contínuas e fornecer uma abordagem diferente da regressão logística.

### Decision Tree

Modelo baseado em regras de decisão, capaz de representar relações não lineares entre as características dos empreendimentos.

---

## Métricas utilizadas

Os modelos foram avaliados utilizando:

- **Acurácia**;
- **Precisão**;
- **Recall**;
- **F1-Score**;
- **Matriz de Confusão**.

Para Precisão, Recall e F1-Score foi utilizada a média **macro**, que calcula a métrica individualmente para cada classe e atribui o mesmo peso às três fontes.

---

## Resultados da classificação

| Modelo | Acurácia | Precisão (macro) | Recall (macro) | F1-Score (macro) |
|---|---:|---:|---:|---:|
| Logistic Regression | 82,47% | 82,82% | 82,14% | 81,97% |
| Gaussian Naive Bayes | 70,23% | 73,32% | 71,52% | 69,94% |
| Decision Tree | **96,13%** | **96,14%** | **96,10%** | **96,12%** |

A **Decision Tree** apresentou os maiores valores nas quatro métricas avaliadas, alcançando **96,13% de acurácia**, **96,14% de precisão macro**, **96,10% de Recall macro** e **96,12% de F1-Score macro**.

Na matriz de confusão, o modelo classificou corretamente **233 registros de Eólica, 286 de Hidráulica e 227 de Solar**. A maior confusão ocorreu entre as classes **Solar e Hidráulica**, com **8 registros solares classificados como hidráulicos**.

---

## Conclusão da classificação

Considerando os resultados obtidos no conjunto de teste, a **Decision Tree** apresentou o melhor desempenho entre os três classificadores avaliados.

Apesar dos resultados, o modelo utiliza apenas `potencia_kw`, `latitude` e `longitude` como entradas. Essas três características não representam todas as diferenças existentes entre as fontes de geração: empreendimentos de fontes distintas podem apresentar potências semelhantes ou estar localizados em regiões próximas.

Por esse motivo, o desempenho obtido neste conjunto de dados não garante o mesmo resultado para novos dados, outras regiões ou diferentes contextos de aplicação.

---

# 2. Regressão da Radiação Solar

## Fonte dos dados

Os dados utilizados nesta etapa são provenientes da **[API histórica Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api)**.

A consulta utiliza dados horários estimados para **Petrolina (PE)**, nas coordenadas aproximadas **-9,39, -40,50**, no período de **01/04/2025 a 30/06/2025**, utilizando o fuso horário `America/Recife`.

Cada registro representa uma hora local entre **7h e 17h**.

Os dados históricos são derivados de modelos e reanálises meteorológicas e não representam medições de geração de um painel fotovoltaico.

---

## Estrutura dos dados

A partir dos dados obtidos pela API, foram preparados os seguintes atributos para a tarefa de regressão:

| Atributo no CSV | Origem na API | Descrição | Papel |
|---|---|---|---|
| `data_hora` | `time` | Data e hora local; use para ordenar e separar por tempo | Identificação, não entrada |
| `temperatura_c` | `temperature_2m` | Temperatura do ar a 2 m, em °C | Entrada |
| `umidade_pct` | `relative_humidity_2m` | Umidade relativa a 2 m, em % | Entrada |
| `nuvens_pct` | `cloud_cover` | Cobertura total de nuvens, em % | Entrada |
| `vento_kmh` | `wind_speed_10m` | Velocidade do vento a 10 m, em km/h | Entrada |
| `hora` | Derivada de `time` | Hora local do registro, de 7 a 17 | Entrada |
| `radiacao_w_m2` | `shortwave_radiation` | Radiação solar global horizontal média da hora anterior, em W/m² | Alvo |

---

## Variáveis utilizadas

Para o treinamento dos modelos de regressão, `X` foi definido por cinco variáveis de entrada:

- `temperatura_c`;
- `umidade_pct`;
- `nuvens_pct`;
- `vento_kmh`;
- `hora`.

A variável alvo `y` corresponde à coluna `radiacao_w_m2`.

A coluna `data_hora` foi utilizada para preservar a ordem cronológica dos registros e realizar a separação temporal entre treino e teste.

Nenhum valor de `radiacao_w_m2`, nem uma transformação direta dessa variável, foi utilizado como entrada do mesmo registro.

---

## Análise inicial dos dados

Antes do treinamento dos modelos foi realizada uma análise exploratória do conjunto de dados.

Foram verificadas:

- quantidade de registros;
- tipos das variáveis;
- presença de valores ausentes;
- estatísticas descritivas;
- correlação entre as variáveis.

O conjunto utilizado possui **1.001 registros** e não apresentou valores ausentes.

Como visualização exploratória, foi utilizada uma matriz de correlação entre as variáveis numéricas. A análise mostrou diferentes níveis de correlação linear com a radiação solar, embora uma correlação linear baixa não signifique necessariamente que uma variável tenha pouca importância para modelos capazes de representar relações não lineares.

---

## Preparação dos dados

Como os registros representam observações ao longo do tempo, a ordem cronológica foi preservada durante a separação entre treino e teste.

Após a ordenação pela coluna `data_hora`, foram utilizados:

- **800 registros iniciais para treinamento**;
- **201 registros finais para teste**.

Dessa forma, aproximadamente 80% das primeiras horas foram utilizadas no treinamento e as 20% finais no teste, sem embaralhamento dos dados.

---

## Modelos de regressão

Foram utilizados três algoritmos diferentes:

### Linear Regression

Modelo linear utilizado como referência para avaliar relações lineares entre as variáveis meteorológicas, a hora do dia e a radiação solar.

### Decision Tree Regressor

Modelo baseado em árvore de decisão, capaz de representar relações não lineares entre as variáveis de entrada e a radiação solar.

### Random Forest Regressor

Modelo que combina várias árvores de decisão, permitindo representar relações não lineares e produzir previsões mais estáveis do que uma única árvore.

---

## Métricas utilizadas

Os modelos foram avaliados utilizando:

- **MAE (Erro Absoluto Médio)** — representa o erro absoluto médio das previsões;
- **MSE (Erro Quadrático Médio)** — penaliza de forma mais intensa erros de maior magnitude;
- **R² (Coeficiente de Determinação)** — indica quanto da variação da variável alvo é explicada pelo modelo.

Para MAE e MSE, valores menores representam erros menores. Para R², valores maiores indicam maior capacidade de explicar a variação da radiação no conjunto avaliado.

---

## Resultados da regressão

| Modelo | MAE (W/m²) | MSE ((W/m²)²) | R² |
|---|---:|---:|---:|
| Linear Regression | 145,20 | 30.034,20 | 0,3598 |
| Decision Tree Regressor | 88,79 | 15.291,16 | 0,6741 |
| Random Forest Regressor | **66,80** | **7.307,42** | **0,8442** |

O **Random Forest Regressor** apresentou o melhor desempenho entre os três modelos avaliados, com o menor MAE, o menor MSE e o maior R².

O MAE de **66,80 W/m²** indica um erro absoluto médio de aproximadamente 66,80 W/m², enquanto o R² de **0,8442** indica que o modelo explicou aproximadamente **84,42% da variação da radiação solar no conjunto de teste**.

Também foi elaborado um gráfico de valores reais × previstos para o Random Forest, permitindo visualizar o comportamento das previsões em relação aos valores reais.

---

## Importância da hora

No Random Forest, a variável `hora` apresentou a maior importância entre as cinco entradas utilizadas, correspondendo a aproximadamente **48,85% da importância calculada pelo modelo**.

Esse resultado reforça a relevância do horário para a estimativa da radiação solar. Como a radiação varia ao longo do dia e sua relação com a hora não é necessariamente linear, uma correlação linear baixa entre `hora` e `radiacao_w_m2` não significa que a variável tenha pouca importância para o modelo.

---

## Conclusão da regressão

Entre os três modelos avaliados, o **Random Forest Regressor** apresentou os melhores resultados no conjunto de teste.

Os resultados sugerem que relações não lineares entre as variáveis meteorológicas, a hora do dia e a radiação solar são relevantes para o problema, considerando o desempenho superior dos modelos baseados em árvores em relação à Regressão Linear.

Entretanto, estimar a radiação solar em W/m² **não equivale automaticamente a prever a energia elétrica produzida por um sistema fotovoltaico**. A geração elétrica também depende de características do sistema, como área e eficiência dos módulos, orientação e inclinação dos painéis, temperatura de operação e perdas.

Portanto, os modelos desenvolvidos nesta etapa estimam a **radiação solar**, e não diretamente a energia elétrica produzida por uma instalação fotovoltaica.

---

# Tecnologias utilizadas

- **Python**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook / Google Colab**
- **Orange Data Mining**
- **APIs REST**
- **Git**
- **GitHub**

---

# Como executar

Clone o repositório:

```bash
git clone https://github.com/JunUehara56/CP5_SERS.git
```

Entre na pasta do projeto:

```bash
cd CP5_SERS
```

O notebook pode ser executado utilizando:

- Google Colab;
- Jupyter Notebook;
- JupyterLab;
- outro ambiente compatível com arquivos `.ipynb`.

Execute as células do notebook em ordem para realizar a obtenção e preparação dos dados, análise exploratória, treinamento dos modelos e avaliação dos resultados.

Os arquivos CSV utilizados no projeto também estão disponíveis no repositório:

- `aneel_classificacao_orange.csv`;
- `meteo_regressao_orange.csv`.

---

# Disciplina

**Soluções em Energias Renováveis e Sustentáveis — SERS**

**Checkpoint 05**

**FIAP — Faculdade de Informática e Administração Paulista**
