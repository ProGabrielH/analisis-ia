import pandas as pd
import kaggle
from langchain_core.tools import tool

kaggle.api.authenticate()


# ── Tools do Interpretador ──────────────────────────────────────────────────

@tool
def ler_colunas_csv(caminho_arquivo: str) -> str:
    """
    Lê um arquivo CSV e retorna:
    - Nomes das colunas
    - Tipos de dados de cada coluna
    - Uma amostra das primeiras 3 linhas (sem índice)
    """
    try:
        print(f"[tool] ler_colunas_csv → {caminho_arquivo}")
        df = pd.read_csv(caminho_arquivo, nrows=5)

        colunas_info = "\n".join(
            f"  - {col} ({dtype})"
            for col, dtype in zip(df.columns, df.dtypes)
        )

        amostra = df.head(3).to_string(index=False)

        return (
            f"Total de colunas: {len(df.columns)}\n\n"
            f"Colunas e tipos:\n{colunas_info}\n\n"
            f"Amostra de dados (3 linhas):\n{amostra}"
        )

    except FileNotFoundError:
        return f"Arquivo não encontrado: {caminho_arquivo}"
    except Exception as e:
        print(f"[tool] Erro: {e}")
        return f"Erro ao ler o arquivo: {str(e)}"


# ── Tools do Analista ───────────────────────────────────────────────────────

@tool
def buscar_datasets_kaggle(tema: str) -> str:
    """
    Busca datasets públicos no Kaggle relacionados ao tema informado.
    Retorna uma lista com nome, autor, link e tamanho dos datasets encontrados.
    """
    try:
        print(f"[tool] buscar_datasets_kaggle → {tema}")
        resultados = kaggle.api.dataset_list(search=tema, max_size=None, page=1)

        if not resultados:
            return f"Nenhum dataset encontrado para o tema: {tema}"

        datasets_formatados = []
        for ds in resultados[:3]:
            datasets_formatados.append(
                f"  **{ds.title}**\n"
                f"   - Autor: {ds._creator_name}\n"
                f"   - Link: https://www.kaggle.com/datasets/{ds.ref}\n"
                f"   - Tamanho: {ds._total_bytes // 1024} KB"
            )

        return "\n".join(datasets_formatados)

    except Exception as e:
        print(f"[tool] Erro: {e}")
        return f"Erro ao buscar datasets: {str(e)}"


# ── Tools do Escritor ───────────────────────────────────────────────────────

@tool
def analisar_csv(caminho_arquivo: str) -> str:
    """
    Realiza uma análise estatística completa de um arquivo CSV e retorna:
    - Estatísticas básicas (média, mediana, desvio padrão, min, max) das colunas numéricas
    - Contagens e distribuições das colunas categóricas e booleanas
    - Correlações entre colunas numéricas
    - Detecção de valores ausentes e outliers (método IQR)
    """
    try:
        print(f"[tool] analisar_csv → {caminho_arquivo}")
        df = pd.read_csv(caminho_arquivo)
        linhas, colunas = df.shape
        secoes = []

        # ── Visão geral ──────────────────────────────────────────────────────
        secoes.append(
            f"## Visão Geral\n"
            f"- Total de linhas: {linhas}\n"
            f"- Total de colunas: {colunas}\n"
            f"- Colunas: {', '.join(df.columns.tolist())}"
        )

        # ── Valores ausentes ─────────────────────────────────────────────────
        ausentes = df.isnull().sum()
        ausentes = ausentes[ausentes > 0]
        if ausentes.empty:
            secoes.append("## Valores Ausentes\nNenhum valor ausente encontrado.")
        else:
            linhas_ausentes = "\n".join(
                f"  - {col}: {qtd} ausentes ({qtd/linhas*100:.1f}%)"
                for col, qtd in ausentes.items()
            )
            secoes.append(f"## Valores Ausentes\n{linhas_ausentes}")

        # ── Estatísticas numéricas ───────────────────────────────────────────
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            stats_linhas = []
            for col in num_cols:
                s = df[col].dropna()
                stats_linhas.append(
                    f"### {col}\n"
                    f"  - Média: {s.mean():.2f} | Mediana: {s.median():.2f} | "
                    f"Desvio padrão: {s.std():.2f}\n"
                    f"  - Mín: {s.min()} | Máx: {s.max()}"
                )
            secoes.append("## Estatísticas Numéricas\n" + "\n".join(stats_linhas))
        else:
            secoes.append("## Estatísticas Numéricas\nNenhuma coluna numérica encontrada.")

        # ── Distribuições categóricas / booleanas ────────────────────────────
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if cat_cols:
            dist_linhas = []
            for col in cat_cols:
                contagem = df[col].value_counts()
                top = "\n".join(
                    f"    - {val}: {cnt} ({cnt/linhas*100:.1f}%)"
                    for val, cnt in contagem.head(10).items()
                )
                dist_linhas.append(f"### {col} ({contagem.nunique()} valores únicos)\n{top}")
            secoes.append("## Distribuições\n" + "\n".join(dist_linhas))

        # ── Correlações ──────────────────────────────────────────────────────
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            pares = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = corr.iloc[i, j]
                    pares.append((num_cols[i], num_cols[j], val))
            pares.sort(key=lambda x: abs(x[2]), reverse=True)
            corr_linhas = "\n".join(
                f"  - {a} × {b}: {v:.2f}"
                for a, b, v in pares
            )
            secoes.append(f"## Correlações\n{corr_linhas}")
        else:
            secoes.append("## Correlações\nColunas numéricas insuficientes para calcular correlações.")

        # ── Outliers (IQR) ───────────────────────────────────────────────────
        if num_cols:
            outlier_linhas = []
            for col in num_cols:
                s = df[col].dropna()
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
                if not outliers.empty:
                    outlier_linhas.append(
                        f"  - {col}: {len(outliers)} outlier(s) "
                        f"(fora do intervalo [{q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}])"
                    )
            if outlier_linhas:
                secoes.append("## Outliers (método IQR)\n" + "\n".join(outlier_linhas))
            else:
                secoes.append("## Outliers (método IQR)\nNenhum outlier detectado.")

        return "\n\n".join(secoes)

    except FileNotFoundError:
        return f"Arquivo não encontrado: {caminho_arquivo}"
    except Exception as e:
        print(f"[tool] Erro: {e}")
        return f"Erro ao analisar o arquivo: {str(e)}"


# ── Tools do Revisor ────────────────────────────────────────────────────────

@tool
def verificar_numeros_relatorio(caminho_arquivo: str, relatorio: str) -> str:
    """
    Verifica se os números presentes no relatório batem com os dados reais do CSV.
    Retorna uma lista de inconsistências encontradas, ou confirmação de que está tudo correto.
    """
    import re

    def parse_numero(texto: str) -> float:
        """
        Converte string numérica para float lidando com separadores de milhar.
        Regra: ponto seguido de exatamente 3 dígitos (sem mais decimais) = milhar.
        Ex: '2.583' → 2583.0, '8.27' → 8.27, '3.15' → 3.15
        """
        texto = texto.strip()
        # Ponto com exatamente 3 dígitos após e nada mais = separador de milhar
        if re.fullmatch(r"\d+\.\d{3}", texto):
            return float(texto.replace(".", ""))
        # Vírgula como separador decimal (ex: '8,27')
        if "," in texto and "." not in texto:
            return float(texto.replace(",", "."))
        return float(texto)

    try:
        print(f"[tool] verificar_numeros_relatorio → {caminho_arquivo}")
        df = pd.read_csv(caminho_arquivo)
        problemas = []

        num_cols = df.select_dtypes(include="number").columns.tolist()
        for col in num_cols:
            s = df[col].dropna()

            checks = {
                f"média.*{col}|{col}.*média": round(s.mean(), 2),
                f"mediana.*{col}|{col}.*mediana": round(s.median(), 2),
                f"mín.*{col}|{col}.*mín": s.min(),
                f"máx.*{col}|{col}.*máx": s.max(),
            }

            for pattern, valor_real in checks.items():
                trechos = re.findall(f".{{0,60}}{pattern}.{{0,60}}", relatorio, re.IGNORECASE)
                matches = re.findall(r"[-+]?\d+[.,]?\d*", " ".join(trechos))
                for m in matches:
                    try:
                        valor_encontrado = parse_numero(m)
                    except ValueError:
                        continue
                    if abs(valor_encontrado - float(valor_real)) > 0.05:
                        problemas.append(
                            f"  - Coluna '{col}': relatório menciona {valor_encontrado}, "
                            f"valor real é {valor_real}"
                        )

        # Verifica total de linhas — só captura em contexto de totalização
        # Padrões: "50.000 estudantes", "total de 50.000 registros", "dataset de 50.000"
        mencoes_linhas = re.findall(
            r"(?:total\s+de|dataset\s+de|amostra\s+de|composta\s+por|analisando|de)\s+([\d][.\d,]*\d|\d+)\s*(?:linhas|registros|entradas|estudantes|alunos)",
            relatorio,
            re.IGNORECASE,
        )
        # Também captura "X estudantes/registros" no início de frase (número grande, >1000)
        mencoes_linhas += re.findall(
            r"(?<!\w)([\d][.\d,]*\d{3,})\s*(?:estudantes|alunos)",
            relatorio,
            re.IGNORECASE,
        )
        for val in mencoes_linhas:
            try:
                valor_int = int(parse_numero(val))
                if valor_int != len(df):
                    problemas.append(
                        f"  - Total de linhas: relatório menciona {valor_int}, arquivo tem {len(df)}"
                    )
            except ValueError:
                continue

        if not problemas:
            return "APROVADO"
        return "Inconsistências encontradas:\n" + "\n".join(problemas)

    except Exception as e:
        print(f"[tool] Erro: {e}")
        return f"Erro ao verificar números: {str(e)}"


# ── Tools do Finalizador ─────────────────────────────────────────────────────

@tool
def salvar_markdown(caminho_saida: str, conteudo: str) -> str:
    """
    Salva o conteúdo do relatório final em um arquivo Markdown (.md).
    Recebe o caminho completo do arquivo de saída e o conteúdo em texto.
    Retorna confirmação com o caminho onde o arquivo foi salvo.
    """
    try:
        print(f"[tool] salvar_markdown → {caminho_saida}")
        if not caminho_saida.endswith(".md"):
            caminho_saida += ".md"

        import os
        os.makedirs(os.path.dirname(caminho_saida) if os.path.dirname(caminho_saida) else ".", exist_ok=True)

        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(conteudo)

        return f"✓ Relatório salvo com sucesso em: {caminho_saida}"

    except Exception as e:
        print(f"[tool] Erro: {e}")
        return f"Erro ao salvar arquivo: {str(e)}"