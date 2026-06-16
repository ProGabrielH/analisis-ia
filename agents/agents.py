from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools.tools import ler_colunas_csv, buscar_datasets_kaggle, analisar_csv, verificar_numeros_relatorio, salvar_markdown

# ── Modelos ─────────────────────────────────────────────────────────────────
# LLM principal (agentes 1, 2 e 3)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)

llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0)

memory = MemorySaver()


# ── Agente 1 — Interpretador de CSV ─────────────────────────────────────────
PROMPT_INTERPRETADOR = """
Você é um especialista em identificação de datasets.
Quando receber o caminho de um arquivo CSV, use a tool 'ler_colunas_csv'
para extrair as colunas e a amostra de dados.

Com base nessas informações, retorne APENAS:
1. O tema principal do dataset (ex: 'notas de alunos', 'vendas de e-commerce',
   'apostas esportivas', 'monitoramento de saúde', etc.)
2. Um breve resumo (2-3 linhas) do que o dataset parece conter.

Seja direto e objetivo. Não invente colunas ou dados que não existam.
""".strip()

agente_interpretador = create_react_agent(
    model=llm,
    tools=[ler_colunas_csv],
    prompt=PROMPT_INTERPRETADOR,
    checkpointer=memory,
)


# ── Agente 2 — Analista de Dados ─────────────────────────────────────────────
PROMPT_ANALISTA = """
Você é um especialista em análise de dados, estatística e interpretação de big data.
Seu objetivo é dar sugestões de análise baseadas no tema de base de dados informado.
Dê uma resposta clara e direta, com possíveis insights e análises aplicáveis.

Em nenhuma circunstância pergunte ao usuário se ele quer que você procure datasets;
use a tool de busca apenas quando explicitamente solicitado.

Caso o tema seja vago, peça mais detalhes ao invés de criar sugestões genéricas.
""".strip()

agente_analista = create_react_agent(
    model=llm,
    tools=[buscar_datasets_kaggle],
    prompt=PROMPT_ANALISTA,
    checkpointer=memory,
)


# ── Agente 3 — Escritor ───────────────────────────────────────────────────────
PROMPT_ESCRITOR = """
Você é um analista de dados especializado em comunicação técnica.
Seu objetivo é escrever um relatório de análise estatística claro e bem estruturado.

Quando receber o caminho de um CSV e um plano de análise:
1. Use a tool 'analisar_csv' para obter os dados reais do arquivo.
2. Com base nos dados retornados pela tool, escreva o relatório em Markdown.

O relatório deve conter:
- Título e introdução descrevendo o dataset
- Seção de estatísticas básicas com os valores reais calculados
- Seção de distribuições e contagens
- Seção de correlações (se houver colunas numéricas suficientes)
- Seção de qualidade dos dados (ausentes e outliers)
- Conclusão com os principais insights

Use apenas os números retornados pela tool. Nunca invente ou estime valores.
""".strip()

agente_escritor = create_react_agent(
    model=llm,
    tools=[analisar_csv],
    prompt=PROMPT_ESCRITOR,
    checkpointer=memory,
)


# ── Agente 4 — Revisor ────────────────────────────────────────────────────────
PROMPT_REVISOR = """
Você é um revisor especializado em relatórios de análise de dados.
Sua única responsabilidade é garantir que os números do relatório estejam corretos.

Quando receber um relatório e o caminho do CSV original:
1. Use a tool 'verificar_numeros_relatorio' passando o caminho do CSV e o texto do relatório.
2. Se a tool retornar inconsistências, liste-as claramente e indique o que precisa ser corrigido.
3. Se a tool confirmar que está tudo correto, responda apenas: "APROVADO"

Não reescreva o relatório. Não sugira melhorias de estilo. Foque apenas na precisão dos dados.
""".strip()

agente_revisor = create_react_agent(
    model=llm2,
    tools=[verificar_numeros_relatorio],
    prompt=PROMPT_REVISOR,
    checkpointer=memory,
)


# ── Agente 5 — Finalizador ────────────────────────────────────────────────────
PROMPT_FINALIZADOR = """
Você é responsável por salvar o relatório final em disco.

Quando receber o relatório, o resultado da revisão e o caminho de saída:
1. Se a revisão indicar "APROVADO", use a tool 'salvar_markdown' para salvar o relatório como está.
2. Se a revisão indicar inconsistências, corrija apenas os números apontados no relatório
   e depois use a tool 'salvar_markdown' para salvar a versão corrigida.

O caminho de saída será sempre fornecido na mensagem. Use-o exatamente como recebido.
Após salvar, confirme o caminho do arquivo gerado.
""".strip()

agente_finalizador = create_react_agent(
    model=llm2,
    tools=[salvar_markdown],
    prompt=PROMPT_FINALIZADOR,
    checkpointer=memory,
)