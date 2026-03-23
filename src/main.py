import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from toon_format import decode, encode

from src.config import AGENT_CONFIG
from src.ingestion import Chunker, FileIngestionHandler, VectorstoreHandler
from src.reasoning.graph import RagAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_stored_fingerprints(manifest_path: Path) -> dict[str, dict[str, int]] | None:
    """Read and validate fingerprints from a TOON manifest file.

    Args:
        manifest_path: File written by :func:`_save_ingestion_manifest`.

    Returns:
        Map of asset-relative paths to ``mtime_ns`` and ``size``, or ``None`` if
        the file is missing, unreadable, or not a valid manifest.
    """
    if not manifest_path.is_file():
        return None
    try:
        text = manifest_path.read_text(encoding="utf-8")
        data = decode(text)
    except OSError:
        return None
    if not isinstance(data, dict) or not(raw := data.get("files")):
        return None
    out: dict[str, dict[str, int]] = {}
    for rel, meta in raw.items():
        if not isinstance(rel, str) or not isinstance(meta, dict):
            continue
        mtime_ns = meta.get("mtime_ns")
        size = meta.get("size")
        out[rel] = {"mtime_ns": mtime_ns, "size": size}
    return dict(sorted(out.items()))


def _save_ingestion_manifest(manifest_path: Path, fingerprints: dict[str, dict[str, int]]) -> None:
    """Persist fingerprints as TOON next to the FAISS store, creating parents if needed.

    Args:
        manifest_path: Target manifest path.
        fingerprints: Current asset file signatures keyed by path relative to ``assets/``.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"files": fingerprints}
    manifest_path.write_text(encode(payload), encoding="utf-8")


def _ingestion_needed(
    manifest_path: Path,
    fingerprints: dict[str, dict[str, int]],
    vectorstore_handler: VectorstoreHandler,
    index_names_from_assets: set[str],
) -> bool:
    """Return whether the full ingestion pipeline should run.

    Ingestion is required when the manifest is absent while there is work to do,
    fingerprints differ from the stored manifest, or a folder that holds assets
    has no matching FAISS index on disk.

    Args:
        manifest_path: Path to the TOON ingestion manifest.
        fingerprints: Live fingerprints from :meth:`FileIngestionHandler.get_asset_file_fingerprints`.
        vectorstore_handler: Used to list existing index names under the FAISS directory.
        index_names_from_assets: Index names implied by the asset tree (parent folder per file).

    Returns:
        ``True`` if chunking and embedding should run; ``False`` to skip.
    """
    existing_indexes = set(vectorstore_handler.list_indexes())
    missing_indexes = index_names_from_assets - existing_indexes

    stored = _load_stored_fingerprints(manifest_path)
    if stored is None:
        return bool(fingerprints) or bool(missing_indexes)

    if stored != fingerprints:
        return True
    return bool(missing_indexes)


def ingest_files_routine() -> None:
	"""Convert assets to markdown, chunk, embed, and save FAISS indexes when needed.

	Skips Docling conversion and embedding when the TOON manifest matches current
	files and all required indexes already exist under ``FAISS_INDEXING_PATH``.
	"""
	file_ingestion_handler = FileIngestionHandler()
	chunker = Chunker()
	vectorstore_handler = VectorstoreHandler(AGENT_CONFIG.EMBEDDING_MODEL, AGENT_CONFIG.FAISS_INDEXING_PATH)

	manifest_path = AGENT_CONFIG.FAISS_INDEXING_PATH / ".ingestion_manifest.toon"
	fingerprints = file_ingestion_handler.get_asset_file_fingerprints()
	index_names = file_ingestion_handler.index_names_for_current_assets()
	if not _ingestion_needed(manifest_path, fingerprints, vectorstore_handler, index_names):
		logger.info("Ingestion skipped: no new indexes and no new or changed asset files.")
		return

	markdown_files = file_ingestion_handler.get_markdown_docs()
	docs_chunks = chunker.get_all_documents_chunks(markdown_files)
	vectorstore_handler.save_chunks_to_vectorstore(docs_chunks)
	_save_ingestion_manifest(manifest_path, fingerprints)


class ChatBot:
	"""Thin CLI wrapper around ``RagAgent``."""

	def __init__(self) -> None:
		"""Build a :class:`RagAgent` instance for interactive chat."""
		self.llm: RagAgent = RagAgent(AGENT_CONFIG.LLM_MODEL_VERSION)

	def process(self, user_input: str) -> str:
		"""Forward ``user_input`` to the agent and return the reply as a string.

		Args:
			user_input: Raw text from the CLI prompt.

		Returns:
			The model response, coerced to ``str`` if needed.
		"""
		result = self.llm.ask(user_input)
		return result if isinstance(result, str) else str(result)


def main() -> None:
	"""Run the Rich CLI loop until the user types ``exit``."""
	ingest_files_routine()

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
	main()