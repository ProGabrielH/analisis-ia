# 📊 Analisis-IA — Sistema Multiagente de Análise Estatística de CSV

Sistema multiagente construído com **LangGraph** e **LangChain** que recebe um arquivo CSV como entrada e gera automaticamente um relatório estatístico em Markdown.

---

## 🤖 Como funciona

O sistema é composto por cinco agentes especializados que operam em sequência:

```
Interpretador → Analista → Escritor → Revisor
                               ↑          │
                               └── Não ───┤ (até 3 tentativas)
                                          │
                                         Sim
                                          ↓
                                      Finalizador
```

| Agente | Responsabilidade |
| :--- | :--- |
| **Interpretador** | Lê as colunas e identifica o tema do dataset |
| **Analista** | Elabora um plano de análise com base no tema |
| **Escritor** | Calcula as estatísticas reais com pandas e redige o relatório |
| **Revisor** | Verifica se os números do relatório batem com os dados do CSV |
| **Finalizador** | Salva o relatório final em disco como `.md` |

Se o Revisor encontrar inconsistências, o relatório volta ao Escritor para correção — até 3 vezes antes de finalizar.

---

## 📁 Estrutura do projeto

```
analisis-ia/
├── agents/
│   └── agents.py       # Definição dos cinco agentes
├── tools/
│   └── tools.py        # Ferramentas utilizadas pelos agentes
├── main.py             # Ponto de entrada e grafo LangGraph
├── requirement.txt     # Dependências do projeto
└── .env.example        # Modelo de variáveis de ambiente
```

---

## ⚙️ Instalação

**1. Clone o repositório**
```bash
git clone https://github.com/ProGabrielH/analisis-ia.git
cd analisis-ia
```

**2. Crie e ative o ambiente virtual**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

**3. Instale as dependências**
```bash
pip install -r requirement.txt
```

**4. Configure as variáveis de ambiente**
```bash
cp .env.example .env
```
Edite o `.env` com suas chaves:
```
GOOGLE_API_KEY=sua_chave_aqui
KAGGLE_USERNAME=seu_usuario_aqui
KAGGLE_KEY=sua_chave_kaggle_aqui
```

**5. Configure o Kaggle**

Baixe o `kaggle.json` em kaggle.com → Settings → API e coloque em:
```
~/.kaggle/kaggle.json
```

---

## 🚀 Uso

```bash
python main.py caminho/para/arquivo.csv
```

O relatório será salvo automaticamente como `<nome_do_arquivo>_analise.md` na mesma pasta do CSV.

---

## 📦 Dependências

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [langchain-google-genai](https://pypi.org/project/langchain-google-genai/)
- [pandas](https://pandas.pydata.org/)
- [kaggle](https://pypi.org/project/kaggle/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## ⚠️ Limitações conhecidas

- Em CSVs com poucos dados numéricos ou tema ambíguo, o sistema pode gerar análises imprecisas
- O sistema opera exclusivamente via terminal
- Utiliza modelos leves (`gemini-3.1-flash-lite`) por restrições de uso gratuito

---

## 👤 Autor

Gabriel Henrique de Araújo Albuquerque
