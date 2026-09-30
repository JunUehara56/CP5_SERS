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

# 1. Classificação (ANEEL)

## Fonte dos dados

Os dados utilizados nesta etapa são provenientes do **[SIGA — Sistema de Informações de Geração da ANEEL](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel)**.

Cada registro do conjunto de dados representa um empreendimento de geração de energia no Brasil.

As fontes foram agrupadas em três classes:

- **Solar** — empreendimentos UFV;
- **Eólica** — empreendimentos EOL;
- **Hidráulica** — empreendimentos UHE, PCH e CGH.

| Atributo no CSV | Origem na API | Descrição | Papel |
|---|---|---|---|
| `potencia_kw` | `MdaPotenciaOutorgadaKw` | Potência outorgada em quilowatts; não representa energia produzida | Entrada |
| `latitude` | `NumCoordNEmpreendimento` | Latitude aproximada, em graus decimais | Entrada |
| `longitude` | `NumCoordEEmpreendimento` | Longitude aproximada, em graus decimais | Entrada |
| `fonte` | `SigTipoGeracao` | Categoria da fonte, agrupada em três classes | Alvo |

Os dados representam informações cadastrais dos empreendimentos e não correspondem à quantidade de energia efetivamente gerada.

---

## Variáveis utilizadas

Para realizar a classificação foram utilizadas três variáveis de entrada:

| Variável | Descrição |
|---|---|
| `potencia_kw` | Potência outorgada do empreendimento em quilowatts |
| `latitude` | Latitude aproximada do empreendimento |
| `longitude` | Longitude aproximada do empreendimento |

A variável alvo utilizada foi:

| Variável | Descrição |
|---|---|
| `fonte` | Tipo de fonte de geração: Solar, Eólica ou Hidráulica |

Dessa forma:

- `X` contém `potencia_kw`, `latitude` e `longitude`;
- `y` contém a coluna `fonte`.

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

- **Accuracy**;
- **Precision**;
- **Recall**;
- **F1-Score**;
- **Matriz de Confusão**.

Para Precision, Recall e F1-Score foi utilizada a média **macro**, atribuindo o mesmo peso para cada uma das três classes.

---

## Resultados da classificação

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|---|---:|---:|---:|---:|
| Logistic Regression | 82,47% | 82,82% | 82,14% | 81,97% |
| Gaussian Naive Bayes | 70,23% | 73,32% | 71,52% | 69,94% |
| Decision Tree | **96,13%** | **96,14%** | **96,10%** | **96,12%** |

A **Decision Tree** apresentou os maiores valores das métricas no conjunto de teste utilizado.

Na matriz de confusão desse modelo foram obtidas **233 classificações corretas para Eólica, 286 para Hidráulica e 227 para Solar**. A maior confusão observada ocorreu entre Solar e Hidráulica, com 8 registros solares classificados como hidráulicos.

---

## Conclusão da classificação

Entre os três classificadores avaliados, a **Decision Tree** apresentou os maiores resultados no conjunto de teste, alcançando aproximadamente **96,13% de acurácia**.

Apesar do desempenho obtido, a classificação utiliza somente potência, latitude e longitude. Essas características não representam todas as diferenças existentes entre fontes de geração.

Empreendimentos de fontes diferentes podem apresentar potências semelhantes ou estar localizados em regiões próximas. Portanto, os resultados obtidos neste conjunto de dados não garantem o mesmo desempenho em novos dados ou em outros contextos.

---

# 2. Regressão da Radiação Solar

> Esta seção será completada após a implementação e avaliação dos modelos de regressão.

---

# Tecnologias utilizadas

- **Python**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook / Google Colab**
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

Os arquivos CSV utilizados no projeto também estão disponíveis no repositório.

---

# Disciplina

**Soluções em Energias Renováveis e Sustentáveis — SERS**

**Checkpoint 05**

**FIAP — Faculdade de Informática e Administração Paulista**
