from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from src.config import AGENT_CONFIG, PROJECT_ROOT
from src.ingestion import Chunker, FileIngestionHandler, VectorstoreHandler
from src.reasoning.graph import RagAgent


class ChatBot:

	def __init__(self):
		self.llm = RagAgent()

	def process(self, user_input: str) -> str:
		return self.llm.ask(user_input)


def ingest_files_routine():
	"""Routine for file reading, chunking, embedding and indexing inside FAISS"""
	file_ingestion_handler = FileIngestionHandler()
	chunker = Chunker()
	vectorstore_handler = VectorstoreHandler(AGENT_CONFIG.EMBEDDING_MODEL, AGENT_CONFIG.FAISS_INDEXING_PATH)

	markdown_files = file_ingestion_handler.get_markdown_docs()
	docs_chunks = chunker.get_all_documents_chunks(markdown_files)
	vectorstore_handler.save_chunks_to_vectorstore(docs_chunks)
	for index in vectorstore_handler.list_indexes():
		print(index)


def main():

	bot = ChatBot()
	console = Console()
	console.print(Panel("Bem-vindo ao [bold cyan]Chat CLI[/bold cyan]! ([green]digite 'sair' para encerrar[/green])", style="bold magenta"))
	while True:
		user_input = Prompt.ask("[bold blue]Você[/bold blue]", console=console)
		if user_input.strip().lower() == 'sair':
			console.print("[yellow]Encerrando o chat. Até logo![/yellow]")
			break
		resposta = bot.process(user_input)
		console.print(Panel(f"[bold white]{resposta}[/bold white]", title="[bold green]Bot[/bold green]", style="green"))

if __name__ == "__main__":
	ingest_files_routine()
	main()