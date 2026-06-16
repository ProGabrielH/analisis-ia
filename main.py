import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.agents import (
    agente_interpretador,
    agente_analista,
    agente_escritor,
    agente_revisor,
    agente_finalizador,
)

MAX_TENTATIVAS = 3

# ── Estado compartilhado entre os agentes ────────────────────────────────────

class MultiAgentState(TypedDict, total=False):
    path: str
    theme: str
    plan: str
    draft: str
    review: str
    final_answer: str
    output_path: str
    tentativas: int          # quantas vezes o escritor já reescreveu


# ── Extração do texto da resposta ────────────────────────────────────────────

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


# ── Nós do grafo ─────────────────────────────────────────────────────────────

def interpreter_node(state: MultiAgentState) -> MultiAgentState:
    print("\n[1/5] Interpretador analisando o CSV...")
    response = agente_interpretador.invoke(
        {"messages": [{"role": "user", "content": f"Faça a interpretação do arquivo CSV: {state['path']}"}]},
        config={"configurable": {"thread_id": "main"}},
    )
    content = extract_text(response["messages"][-1].content)
    print(f"  → Tema identificado: {content[:120]}...")
    return {"theme": content}


def analyst_node(state: MultiAgentState) -> MultiAgentState:
    print("\n[2/5] Analista elaborando plano de análise...")
    response = agente_analista.invoke(
        {"messages": [{"role": "user", "content": f"Faça a análise do tema: {state['theme']}"}]},
        config={"configurable": {"thread_id": "main"}},
    )
    content = extract_text(response["messages"][-1].content)
    print(f"  → Plano gerado: {content[:120]}...")
    return {"plan": content}


def writer_node(state: MultiAgentState) -> MultiAgentState:
    tentativas = state.get("tentativas", 0) + 1
    review = state.get("review", "")

    if tentativas == 1:
        print("\n[3/5] Escritor redigindo a análise...")
        prompt = (
            f"Escreva uma análise estatística dos dados em: {state['path']}\n\n"
            f"Siga este plano:\n{state['plan']}"
        )
    else:
        print(f"\n[3/5] Escritor corrigindo (tentativa {tentativas}/{MAX_TENTATIVAS})...")
        prompt = (
            f"Reescreva a análise abaixo corrigindo as inconsistências apontadas pelo revisor.\n\n"
            f"Inconsistências encontradas:\n{review}\n\n"
            f"Relatório atual:\n{state['draft']}\n\n"
            f"Use a tool 'analisar_csv' com o arquivo {state['path']} para confirmar os valores corretos."
        )

    response = agente_escritor.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": "main"}},
    )
    content = extract_text(response["messages"][-1].content)
    print(f"  → Rascunho gerado ({len(content)} chars)")
    return {"draft": content, "tentativas": tentativas}


def reviewer_node(state: MultiAgentState) -> MultiAgentState:
    print("\n[4/5] Revisor verificando números do relatório...")
    response = agente_revisor.invoke(
        {"messages": [{
            "role": "user",
            "content": (
                f"Verifique se os números do relatório abaixo batem com o CSV em: {state['path']}\n\n"
                f"Relatório:\n{state['draft']}"
            ),
        }]},
        config={"configurable": {"thread_id": "main"}},
    )
    content = extract_text(response["messages"][-1].content)
    print(f"  → Revisão: {content[:120]}...")
    return {"review": content}


def finalizer_node(state: MultiAgentState) -> MultiAgentState:
    print("\n[5/5] Finalizador salvando o relatório...")
    nome_base = os.path.splitext(os.path.basename(state["path"]))[0]
    output_path = os.path.join(os.path.dirname(state["path"]), f"{nome_base}_analise.md")

    response = agente_finalizador.invoke(
        {"messages": [{
            "role": "user",
            "content": (
                f"Salve o relatório final no caminho: {output_path}\n\n"
                f"Relatório:\n{state['draft']}"
            ),
        }]},
        config={"configurable": {"thread_id": "main"}},
    )
    content = extract_text(response["messages"][-1].content)
    print(f"  → {content}")
    return {"final_answer": state["draft"], "output_path": output_path}


# ── Edge condicional: revisor → escritor ou finalizador ──────────────────────

def deve_reescrever(state: MultiAgentState) -> str:
    review = state.get("review", "").upper()
    tentativas = state.get("tentativas", 0)

    aprovado = "APROVADO" in review

    if aprovado:
        print("  ✓ Revisor aprovou — seguindo para finalização.")
        return "finalizer"

    if tentativas >= MAX_TENTATIVAS:
        print(f"  ⚠ Limite de {MAX_TENTATIVAS} tentativas atingido — finalizando mesmo assim.")
        return "finalizer"

    print(f"  ✗ Revisor reprovou — devolvendo para o escritor (tentativa {tentativas}/{MAX_TENTATIVAS}).")
    return "writer"


# ── Construção do grafo ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(MultiAgentState)

    graph.add_node("interpreter", interpreter_node)
    graph.add_node("analyst",     analyst_node)
    graph.add_node("writer",      writer_node)
    graph.add_node("reviewer",    reviewer_node)
    graph.add_node("finalizer",   finalizer_node)

    graph.set_entry_point("interpreter")
    graph.add_edge("interpreter", "analyst")
    graph.add_edge("analyst",     "writer")
    graph.add_edge("writer",      "reviewer")
    graph.add_conditional_edges(   # ← edge condicional
        "reviewer",
        deve_reescrever,
        {"writer": "writer", "finalizer": "finalizer"},
    )
    graph.add_edge("finalizer", END)

    return graph.compile()


# ── Ponto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python main.py <caminho_do_arquivo.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Arquivo não encontrado: {csv_path}")
        sys.exit(1)

    app = build_graph()

    print(f"\n{'='*60}")
    print(f"  Sistema Multiagente — Análise de CSV")
    print(f"  Arquivo: {csv_path}")
    print(f"{'='*60}")

    result = app.invoke({"path": csv_path})

    print(f"\n{'='*60}")
    print("  RELATÓRIO FINAL")
    print(f"{'='*60}\n")
    print(result["final_answer"])
    print(f"\n{'='*60}")
    print(f"  Relatório salvo em: {result.get('output_path', 'caminho não registrado')}")
    print(f"{'='*60}")