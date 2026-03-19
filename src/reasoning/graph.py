from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.retrieval import RetrievalHandler

from prompts import CONTEXTO_ORGANIZACIONAL_PROMPT, OWASP_ASSISTANT_PROMPT, ROUTER_PROMPT

from config import AGENT_CONFIG, PROJECT_ROOT

class SecurityAgentState(MessagesState):
    intent: str

class SecurityAgent:
    def __init__(self, model: str = "claude-haiku-4-5"):
        self.llm = init_chat_model(model=model)
        self.thread = { "configurable": { "thread_id": "fixed" }}
        self.tools = [RetrievalHandler(AGENT_CONFIG.EMBEDDING_MODEL, PROJECT_ROOT).query_vectorstore]
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
        def assistant_node(state: SecurityAgentState) -> SecurityAgentState:
            response = chain.invoke({ "messages": state["messages"]})
            return { "messages": [response] } 
        return assistant_node
    
    # CHAMADOR DE FERRAMENTAS
    def make_tool_caller_node(self, system_prompt):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{messages}")
        ])
        chain = ( prompt | self.llm.bind_tools(self.tools) )

        def tool_caller_node(state: SecurityAgentState) -> SecurityAgentState:
            response = chain.invoke({ "messages": state["messages"]})
            return { "messages": [response] } 
        return tool_caller_node
    
    # DETECTOR DE INTENCAO
    def make_intent_router_node(self, system_prompt):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        chain = ( prompt | self.llm | StrOutputParser() )
        
        def intent_router_node(state: SecurityAgentState) -> SecurityAgentState:
            answer = chain.invoke({ "input": state["messages"][-1]})

            return { "intent": answer }
        
        return intent_router_node
    
    # INTENT CONDITION
    def make_intent_condition(self):
        def intent_condition(state: SecurityAgentState) -> str:
            if state["intent"] == "organization":
                return "org_assistant"
            if state["intent"] == "security":
                return "owasp_assistant"
            else:
                return END
        return intent_condition
    
    def build_graph(self):
        # BUILDERS
        router_node = self.make_intent_router_node(ROUTER_PROMPT)
        org_node = self.make_conversation_node(CONTEXTO_ORGANIZACIONAL_PROMPT)
        owasp_node = self.make_tool_caller_node(OWASP_ASSISTANT_PROMPT)
        intent_condition = self.make_intent_condition()

        builder = StateGraph(SecurityAgentState)

        # NODES
        builder.add_node("router", router_node)
        builder.add_node("org_assistant", org_node)
        builder.add_node("owasp_assistant", owasp_node)
        builder.add_node("tools", ToolNode(self.tools))

        # EDGES
        builder.add_edge(START, "router")
        builder.add_conditional_edges("router", intent_condition, { "org_assistant": "org_assistant", "owasp_assistant": "owasp_assistant", "__end__": END })
        builder.add_conditional_edges("owasp_assistant", tools_condition)
        builder.add_edge("tools", "owasp_assistant")
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