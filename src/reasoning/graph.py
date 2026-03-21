import yaml
from datetime import datetime
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.ingestion import VectorstoreHandler
from src.retrieval import RetrievalHandler

from src.reasoning.prompts import (
    ORGANIZATIONAL_CONTEXT_PROMP,
    ROUTER_PROMPT,
)

from src.config import AGENT_CONFIG, PROJECT_ROOT

class RagAgentState(MessagesState):
    intent: str

class RagAgent:
    def __init__(self, model: str = "claude-haiku-4-5"):
        self.llm = init_chat_model(model=model)
        self.thread = { "configurable": { "thread_id": "fixed" }}
        self.graph = self.build_graph()

    def save_graph_schema(self, graph):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"graph_{timestamp}.mmd"
        mermaid_code = graph.get_graph(xray=True).draw_mermaid()
        with open(file_path, "w") as f:
            f.write(mermaid_code)

    # CONVERSADOR
    def make_conversation_node(self, system_prompt):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{messages}")
        ])
        chain = ( prompt | self.llm )
        def assistant_node(state: RagAgentState) -> RagAgentState:
            response = chain.invoke({ "messages": state["messages"]})
            return { "messages": [response] } 
        return assistant_node
    
    # CHAMADOR DE FERRAMENTAS
    def make_tool_caller_node(self, system_prompt, tools):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{messages}")
        ])
        chain = ( prompt | self.llm.bind_tools(tools) )

        def tool_caller_node(state: RagAgentState) -> RagAgentState:
            response = chain.invoke({ "messages": state["messages"]})
            return { "messages": [response] } 
        return tool_caller_node
    
    # DETECTOR DE INTENCAO
    def make_intent_router_node(self, system_prompt: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        chain = ( prompt | self.llm | StrOutputParser() )
        
        def intent_router_node(state: RagAgentState) -> RagAgentState:
            answer = chain.invoke({ "input": state["messages"][-1]})

            return { "intent": answer }
        
        return intent_router_node
    
    # INTENT CONDITION
    def make_intent_condition(self, intent_options: list[str, str]):
        def intent_condition(state: RagAgentState) -> str:
            for intent, node_name in intent_options:
                if state["intent"] == intent:
                    return node_name
            if state["intent"] == "organization":
                return "org_assistant"
            else:
                return END
        return intent_condition
    
    @staticmethod
    def load_prompt_config(folder_path: str | Path) -> str:
        """Read prompt file and return the string content"""
        prompt_file = Path(folder_path) / "prompt.yaml"
        
        if not prompt_file.exists():
            return None
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def build_graph(self):
        # BUILDERS
        org_node = self.make_conversation_node(ORGANIZATIONAL_CONTEXT_PROMP)

        assistent_nodes_mapping = {}
        tools = []
        router_options = []
        for index in VectorstoreHandler(AGENT_CONFIG.EMBEDDING_MODEL, AGENT_CONFIG.FAISS_INDEXING_PATH).list_indexes():
            retrieval_handler = RetrievalHandler(AGENT_CONFIG.EMBEDDING_MODEL, AGENT_CONFIG.FAISS_INDEXING_PATH, index_name=index)
            prompt = self.load_prompt_config(AGENT_CONFIG.ASSETS_PATH / index)
            node_name = f"{index.lower()}_assistant"
            assistent_nodes_mapping[node_name] = self.make_tool_caller_node(
                prompt["system"],
                [retrieval_handler.query_vectorstore],
            )
            tools.append(retrieval_handler.query_vectorstore)
            router_options.append((prompt["intent"].strip(), node_name))

        intent_options_str = " - ".join(i + "\n" for i, _ in router_options)
        prompt = ROUTER_PROMPT.format(intent_options=intent_options_str)
        router_node = self.make_intent_router_node(prompt)
        intent_condition = self.make_intent_condition(router_options)

        builder = StateGraph(RagAgentState)

        # NODES
        builder.add_node("router", router_node)
        builder.add_node("org_assistant", org_node)
        [builder.add_node(node, action) for node, action in assistent_nodes_mapping.items()]
        builder.add_node("tools", ToolNode(tools))

        # EDGES
        router_edges = {key:key for key in assistent_nodes_mapping}
        router_edges["org_assistant"] = "org_assistant"
        router_edges["__end__"] = END

        builder.add_edge(START, "router")
        builder.add_conditional_edges("router", intent_condition, router_edges)
        for key in assistent_nodes_mapping:
            builder.add_conditional_edges(key, tools_condition)
            builder.add_edge("tools", key)
        builder.add_edge("org_assistant", END)


        # COMPILATION
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)

        self.save_graph_schema(graph)
        return graph

    def ask(self, prompt: str):
        initial_state = { "messages": [ prompt ]}
        response = self.graph.invoke(input=initial_state, config=self.thread)

        return response["messages"][-1].content