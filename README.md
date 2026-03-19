# multi-index-rag-agent
Rag agent developed with lang-graph capable of chosing the correct vectore store index during execution

├── src/
│   ├── main.py           # Aplicação principal
│   ├── vectorstore.py    # Ingestão e geração de embeddings (FAISS)

# 📚 RAG OWASP LLM – Aplicação de Consulta RAG

Este projeto implementa uma aplicação de consulta baseada em RAG (Retrieval-Augmented Generation) utilizando documentos do **OWASP Top 10 for LLM Applications**.

A aplicação permite:
- Ingestão de documentos
- Geração de embeddings
- Armazenamento vetorial com FAISS
- Consulta via aplicação principal

## 📦 Instalação das dependências

Com o Poetry instalado, execute:

```bash
poetry install
```

🧠 Ingestão de documentos e geração de embeddings

```bash
poetry run python ./src/vectorstore.py
```

🚀 Executando a aplicação principal
```bash
poetry run python ./src/main.py
```

## 🧪 Exercício — RAG Multi-Documento com Roteamento Dinâmico

### Objetivo

Expandir a aplicação atual para suportar **ingestão e consulta de múltiplos documentos**, criando um novo fluxo completo de RAG a partir de um arquivo diferente do OWASP.

O exercício envolve:
- ingestão com Docling
- criação de um novo vectorstore FAISS
- adaptação do grafo de agentes para rotear consultas corretamente

---

## 📥 Parte 1 — Nova ingestão de documento

Crie uma nova rota ou fluxo de ingestão que aceite **um arquivo diferente**, podendo ser de um dos seguintes formatos:

- `.pdf`
- `.docx`
- `.csv`
- `.xlsx` / `.xls`
- `.txt`

### Requisitos

- Utilize **Docling** para carregar o arquivo, independentemente do formato.
- Não é necessário criar uma estrutura semântica nova (headers, seções, etc).
- O documento deve ser tratado como **conteúdo bruto**, apenas convertido em texto.

---

## 🧠 Parte 2 — Chunking e novo Vector Store

Após carregar o documento:

1. Aplique o chunking usando LangChain

2. Crie um **novo banco vetorial FAISS**, separado do original.

### Requisitos obrigatórios

- O novo índice deve ter um nome diferente, por exemplo:
  - `faiss_index_2`
- O índice original **não pode ser sobrescrito**
- Cada vectorstore deve ser independente

Exemplo conceitual:
```text
faiss_index       -> OWASP Top 10 LLM
faiss_index_2     -> Novo documento
```

### 🔁 Parte 3 — Adaptação do grafo de agentes


Agora adapte o grafo (LangGraph) para suportar o novo fluxo.

#### DICA IMPORTANTE

Nosso builder de nodos de tools é generico, uma dica é adicionar o array de ferramentas como parametro no builder:

```python
    # CHAMADOR DE FERRAMENTAS
    def make_tool_caller_node(self, system_prompt, tools):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{messages}")
        ])
        chain = ( prompt | self.llm.bind_tools(tools) )

        def tool_caller_node(state: SecurityAgentState) -> SecurityAgentState:
            response = chain.invoke({ "messages": state["messages"]})
            return { "messages": [response] } 
        return tool_caller_node
```
Ai no builder de nodos voce adaptar:
```python
    def build_graph(self): 
        owasp_node = self.make_tool_caller_node(OWASP_ASSISTANT_PROMPT, [get_relevant_docs_owasp])
        new_node = self.make_tool_caller_node(SEU_PROMPT, [get_relevant_docs_new_doc])
```

Assim voce pode passar a lista contendo a nova ferramenta que acessa o vectorstore novo, ao inves de misturar as duas ferramentas, com isso voce pode criar um nodo pra cada vectorstore.

Adaptar o Router Node para identificar o novo fluxo.

Usuário
  ↓
Router Node
  ├── OWASP RAG Node → faiss_index
  ├── Novo Documento RAG Node → faiss_index_2
  ├── Organizational Node
  └── End (fora do escopo)



