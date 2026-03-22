from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.config import AGENT_CONFIG
from src.ingestion import Chunker, FileIngestionHandler, VectorstoreHandler
from src.reasoning.graph import RagAgent


class ChatBot:
	"""Thin CLI wrapper around ``RagAgent``."""

	def __init__(self) -> None:
		self.llm: RagAgent = RagAgent()

	def process(self, user_input: str) -> str:
		result = self.llm.ask(user_input)
		if isinstance(result, str):
			return result
		return str(result)


def ingest_files_routine() -> None:
	"""Routine for file reading, chunking, embedding and indexing inside FAISS"""
	file_ingestion_handler = FileIngestionHandler()
	chunker = Chunker()
	vectorstore_handler = VectorstoreHandler(AGENT_CONFIG.EMBEDDING_MODEL, AGENT_CONFIG.FAISS_INDEXING_PATH)

	markdown_files = file_ingestion_handler.get_markdown_docs()
	docs_chunks = chunker.get_all_documents_chunks(markdown_files)
	vectorstore_handler.save_chunks_to_vectorstore(docs_chunks)


def main() -> None:

	bot = ChatBot()
	console = Console()
	console.print(Panel("Welcome to [bold cyan]Chat CLI[/bold cyan]! ([green]type 'exit' to quit[/green])", style="bold magenta"))
	while True:
		user_input = Prompt.ask("[bold blue]You[/bold blue]", console=console)
		if user_input.strip().lower() == "exit":
			console.print("[yellow]Closing the chat. Goodbye![/yellow]")
			break
		response = bot.process(user_input)
		console.print(Panel(f"[bold white]{response}[/bold white]", title="[bold green]Bot[/bold green]", style="green"))

if __name__ == "__main__":
	ingest_files_routine()
	main()