# 🌍 O Futuro do Trabalho — Predição de Prosperidade Regional com Machine Learning

## 📘 Enunciado do Problema
O avanço da **Inteligência Artificial**, da **automação** e das **mudanças sociais** está transformando profundamente o mundo do trabalho.  
Profissões estão sendo reinventadas e novas habilidades estão se tornando essenciais, como pensamento crítico, criatividade e capacidade analítica.

Com base nisso, este projeto propõe uma análise preditiva para explorar **como fatores socioeconômicos e estruturais podem se relacionar com a prosperidade de uma região**, conceito que pode ser associado à **qualidade de vida, oportunidades de emprego e condições de trabalho**.

---

## 📊 Escolha do Conjunto de Dados
Foi utilizado o **California Housing Dataset**, disponibilizado pela biblioteca `scikit-learn`.  
Esse conjunto de dados contém informações sobre diferentes regiões da Califórnia (EUA), com atributos como:
- **Renda média das famílias**
- **Idade média das casas**
- **Número médio de quartos e moradores por domicílio**
- **Latitude e longitude (localização)**

O objetivo é prever o **valor médio das habitações**, utilizado aqui como **indicador de prosperidade socioeconômica**, refletindo o acesso a oportunidades e qualidade de vida — diretamente conectado ao tema **“O Futuro do Trabalho”**.

---

## 💡 Especificação da Solução Proposta

A solução desenvolvida é uma **aplicação Python** que:
1. Carrega e prepara os dados reais (California Housing);
2. Aplica **normalização** dos atributos para melhorar o desempenho dos algoritmos;
3. Treina e avalia **três modelos de regressão**:
   - **LinearRegression** (modelo base);
   - **Ridge Regression (L2)**, que reduz o impacto de atributos com pesos muito altos;
   - **Lasso Regression (L1)**, que elimina atributos irrelevantes e realiza seleção automática de variáveis;
4. Executa **validação cruzada (5-fold)** para obter uma métrica de desempenho mais estável e confiável;
5. Exibe os resultados comparativos dos modelos, incluindo **R²**, **MSE** e **médias da validação cruzada**;
6. Mostra **gráficos interativos** com:
   - Comparação do desempenho entre modelos;
   - Relação entre valores reais e preditos do melhor modelo.

---

## 🧠 Estratégia Utilizada
1. **Preparação dos dados:** normalização com `StandardScaler`.  
2. **Treinamento e Validação:** utilização de três modelos com técnicas de regularização (Ridge e Lasso).  
3. **Validação Cruzada:** `cross_val_score(cv=5)` para medir a estabilidade do modelo.  
4. **Avaliação:** métricas **R²** (qualidade da predição) e **MSE** (erro médio).  
5. **Visualização:** gráficos para análise comparativa e desempenho final.

---

## ⚙️ Justificativa das Ferramentas e Modelos
| Ferramenta / Modelo | Justificativa |
|----------------------|----------------|
| **Python** | Linguagem amplamente utilizada em ciência de dados e IA. |
| **scikit-learn** | Biblioteca robusta para modelagem, regressão, validação e métricas. |
| **pandas / numpy** | Manipulação de dados numéricos e tabelares. |
| **matplotlib** | Geração de gráficos e visualização dos resultados. |
| **LinearRegression** | Modelo interpretável, usado como baseline. |
| **Ridge (L2)** | Reduz overfitting ao penalizar coeficientes grandes. |
| **Lasso (L1)** | Realiza regularização e seleção de variáveis. |
| **Validação Cruzada** | Melhora a confiabilidade e reduz viés na avaliação. |

---

## 📈 Análises e Resultados

Os resultados mostraram que:
- A **regularização** (Ridge e Lasso) melhorou a **estabilidade dos modelos** em relação à regressão linear simples.  
- O **Ridge Regression** apresentou o melhor equilíbrio entre erro e poder de explicação (R²).  
- O **R² médio** ficou entre **0.58 e 0.62**, o que é típico desse conjunto de dados.  
- A abordagem permite observar a importância de variáveis como **renda média**, **densidade populacional** e **localização geográfica**, todas relacionadas à qualidade de vida e, indiretamente, às **oportunidades de trabalho**.

---

## 🧩 Conclusão

O projeto demonstra como **técnicas de aprendizado de máquina** podem ser aplicadas para compreender **relações entre desenvolvimento socioeconômico e condições de trabalho futuras**.  
A solução proposta utiliza **modelos interpretáveis**, combinados com **validação cruzada e regularização**, garantindo resultados consistentes e explicáveis.

Essa abordagem ilustra o potencial da **educação tecnológica e da análise de dados** como ferramentas para construir **um futuro do trabalho mais inclusivo, ético e sustentável**, alinhado à visão da **ONU** e **OIT** para 2030–2050.

---

## 🧰 Requirements

### 🖥️ Requisitos do Sistema
- Python **3.8+**
- 4 GB de RAM (mínimo recomendado)
- Sistema operacional: Windows, macOS ou Linux

### 📦 Dependências do Projeto
Instale todas as bibliotecas com o comando:
```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 Integrantes
* RM553377 - Enzo Rodrigues
* RM553266 - Hugo Santos
* RM553521 - Rafael Cristofali
