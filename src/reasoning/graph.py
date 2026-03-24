from __future__ import annotations

import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from src.ingestion import VectorstoreHandler
from src.retrieval import RetrievalHandler

from src.reasoning.prompts import (
    ORGANIZATIONAL_CONTEXT_PROMP,
    GENERAL_POURPUSE_PROMPT,
    ROUTER_PROMPT,
)

from src.config import AGENT_CONFIG

StatePatch = dict[str, Any]


class RagAgentState(MessagesState):
    """Shared LangGraph state: chat history plus the router's intent label."""

    intent: str

class RagAgent:
    """Multi-index RAG agent: routes by intent, then runs domain assistants with retrieval tools."""

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        """Initialize the chat model, thread config, and compiled graph.

        Args:
            model: Model identifier passed to LangChain's ``init_chat_model``.
        """
        self.llm: BaseChatModel = init_chat_model(model=model)
        self.thread: dict[str, dict[str, str]] = {
            "configurable": {"thread_id": "fixed"},
        }
        self.graph: CompiledStateGraph[RagAgentState, Any, Any, Any] = self._build_graph()

    @staticmethod
    def load_prompt_config(folder_path: str | Path) -> dict[str, Any] | None:
        """Load ``prompt.yaml`` for a FAISS index asset folder.

        Args:
            folder_path: Directory containing ``prompt.yaml`` (e.g. asset index name).

        Returns:
            Parsed YAML as a dict (e.g. ``system``, ``intent``,
            ``classification_prompt``), or ``None`` if the file is missing.
        """
        prompt_file = Path(folder_path) / "prompt.yaml"
        
        if not prompt_file.exists():
            return None
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        return data

    def _save_graph_schema(self, graph: CompiledStateGraph[RagAgentState, Any, Any, Any]) -> None:
        """Persist a Mermaid diagram of the compiled graph next to the project root.

        Args:
            graph: A compiled LangGraph instance with ``get_graph(...).draw_mermaid()``.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"graph_{timestamp}.mmd"
        mermaid_code = graph.get_graph(xray=True).draw_mermaid()
        with open(file_path, "w") as f:
            f.write(mermaid_code)

    def _make_conversation_node(
        self,
        system_prompt: str,
    ) -> Callable[[RagAgentState], StatePatch]:
        """Build a graph node that answers from chat history without tools.

        Used for the organizational assistant: system instructions plus the full
        ``messages`` list.

        Args:
            system_prompt: System message content for the chat template.

        Returns:
            Callable that maps ``RagAgentState`` to an update containing the new
            assistant ``AIMessage``.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{messages}")
        ])
        chain = ( prompt | self.llm )
        def assistant_node(state: RagAgentState) -> StatePatch:
            """Invoke the plain chat chain on ``state["messages"]``."""
            response = chain.invoke({"messages": state["messages"]})
            return {"messages": [response]}
        return assistant_node

    def _make_tool_caller_node(
        self,
        system_prompt: str,
        tools: list[BaseTool | Callable[..., Any]],
    ) -> Callable[[RagAgentState], StatePatch]:
        """Build a graph node that may call the given tools (e.g. vector retrieval).

        The LLM is bound only to ``tools``, so each domain assistant can use a
        dedicated retrieval function without seeing other indexes' tools.

        Args:
            system_prompt: System message for this domain assistant.
            tools: Sequence of LangChain tools (typically one retriever per index).

        Returns:
            Callable that maps ``RagAgentState`` to an update with the model
            response (plain reply or tool calls).
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{messages}")
        ])
        chain = ( prompt | self.llm.bind_tools(tools) )

        def tool_caller_node(state: RagAgentState) -> StatePatch:
            """Invoke the tool-bound model on ``state["messages"]``."""
            response = chain.invoke({"messages": state["messages"]})
            return {"messages": [response]}
        return tool_caller_node

    def _make_intent_router_node(
        self,
        system_prompt: str,
    ) -> Callable[[RagAgentState], StatePatch]:
        """Build the router node: classifies the latest user turn into an intent string.

        The system prompt (built from ``ROUTER_PROMPT``) lists valid intent labels
        and criteria. The model must output a single token/label consumed by
        ``_make_intent_condition``.

        Args:
            system_prompt: Router instructions including intent list and rules.

        Returns:
            Callable that updates state with ``{"intent": <label>}``.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        chain = ( prompt | self.llm | StrOutputParser() )
        
        def intent_router_node(state: RagAgentState) -> StatePatch:
            """Set ``intent`` from the latest message only."""
            answer = chain.invoke({"input": state["messages"][-1]})

            return {"intent": answer}

        return intent_router_node

    def _make_intent_condition(
        self,
        intent_options: list[tuple[str, str, str]],
    ) -> Callable[[RagAgentState], str]:
        """Build the routing function used after the router node.

        Maps the string in ``state["intent"]`` to the next graph node name, or
        ``END`` when no domain matches.

        Args:
            intent_options: Tuples ``(intent_label, classification_block, node_name)``
                as built for ``ROUTER_PROMPT``; only ``intent_label`` and
                ``node_name`` are used here.

        Returns:
            Callable compatible with ``StateGraph.add_conditional_edges`` from
            ``router``, returning a successor node key or ``END``.
        """
        def intent_condition(state: RagAgentState) -> str:
            """Return the assistant node name for ``state["intent"]``."""
            for intent, _, node_name in intent_options:
                if state["intent"] == intent:
                    return node_name
            if state["intent"] == "organization":
                return "org_assistant"
            else:
                return "general_assistant"
        return intent_condition

    def _build_graph(self) -> CompiledStateGraph[RagAgentState, Any, Any, Any]:
        """Assemble and compile the LangGraph workflow.

        Flow:

        - ``START`` → ``router`` (intent from last message).
        - Router → ``org_assistant`` or a per-index ``*_assistant`` node, or ``END``.
        - Each RAG assistant has its own ``*_assistant_tools`` ``ToolNode`` (single
          tool) to avoid fan-out to other domains after tool execution.
        - Organizational path ends after one model turn.

        Side effects:
            Writes a timestamped ``graph_*.mmd`` file via ``_save_graph_schema``.

        Returns:
            Compiled state graph with in-memory checkpointing.
        """
        # BUILDERS
        org_node = self._make_conversation_node(ORGANIZATIONAL_CONTEXT_PROMP)
        general_node = self._make_conversation_node(GENERAL_POURPUSE_PROMPT)

        assistent_nodes_mapping: dict = {}
        router_options: list[tuple[str, str, str]] = []
        indexes = VectorstoreHandler(
            AGENT_CONFIG.EMBEDDING_MODEL,
            AGENT_CONFIG.FAISS_INDEXING_PATH,
        ).list_indexes()
        for enum_index, faiss_index in enumerate(indexes, start=2):
            retrieval_handler = RetrievalHandler(
                AGENT_CONFIG.EMBEDDING_MODEL,
                AGENT_CONFIG.FAISS_INDEXING_PATH,
                index_name=faiss_index,
            )
            prompt = self.load_prompt_config(AGENT_CONFIG.ASSETS_PATH / faiss_index)
            if prompt is None:
                raise FileNotFoundError(
                    f"Missing or invalid prompt.yaml for index {faiss_index!r} under "
                    f"{AGENT_CONFIG.ASSETS_PATH}"
                )
            node_name = f"{faiss_index.lower()}_assistant"
            tools_node_name = f"{node_name}_tools"
            tool_caller_node = self._make_tool_caller_node(
                prompt["system"],
                [retrieval_handler.query_vectorstore],
            )
            assistent_nodes_mapping[node_name] = {
                "tool_node": tool_caller_node,
                "tools_node_name": tools_node_name,
                "tool": retrieval_handler.query_vectorstore,
            }
            router_options.append(
                (
                    prompt["intent"].strip(),
                    str(enum_index) + prompt["classification_prompt"].strip(),
                    node_name,
                )
            )

        intent_options_str = " - ".join(i + "\n" for i, _, _ in router_options)
        classification_prompts = "\n\n".join(i for _, i, _ in router_options)
        router_system_prompt = ROUTER_PROMPT.format(
            intent_options=intent_options_str,
            additional_classification_criterias=classification_prompts,
        )

        router_node = self._make_intent_router_node(router_system_prompt)
        intent_condition = self._make_intent_condition(router_options)

        builder = StateGraph(RagAgentState)

        # NODES
        builder.add_node("router", router_node)
        builder.add_node("org_assistant", org_node)
        builder.add_node("general_assistant", general_node)
        for node_name, tool_map in assistent_nodes_mapping.items():
            builder.add_node(node_name, tool_map["tool_node"])
            builder.add_node(
                tool_map["tools_node_name"],
                ToolNode([tool_map["tool"]]),
            )

        # EDGES
        router_edges = {key:key for key in assistent_nodes_mapping}
        router_edges["org_assistant"] = "org_assistant"
        router_edges["general_assistant"] = "general_assistant"
        
        builder.add_edge(START, "router")
        builder.add_conditional_edges("router", intent_condition, router_edges)
        for key, tool_map in assistent_nodes_mapping.items():
            tools_node = tool_map["tools_node_name"]
            builder.add_conditional_edges(
                key,
                tools_condition,
                {"tools": tools_node, "__end__": END},
            )
            builder.add_edge(tools_node, key)
        builder.add_edge("org_assistant", END)
        builder.add_edge("general_assistant", END)

        # COMPILATION
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)

        self._save_graph_schema(graph)
        return graph

    def ask(self, prompt: str) -> Any:
        """Run the graph on a user string and return the last assistant message text.

        Uses the fixed ``thread_id`` in ``self.thread`` so checkpointed state
        persists across calls on the same ``RagAgent`` instance.

        Args:
            prompt: User message (string); stored as the latest human message.

        Returns:
            ``content`` of the final message in the thread after the run (often a
            string; may be structured for multimodal models).
        """
        initial_state = { "messages": [ prompt ]}
        response = self.graph.invoke(input=initial_state, config=self.thread)

        return response["messages"][-1].content