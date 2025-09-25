"""Gradio user interface for the RAG-Anything pipeline."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import gradio as gr
from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig


# Load optional environment variables from a .env file if present
load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"), override=False)

# Default configuration values pulled from the environment when available
DEFAULT_WORKING_DIR = Path(os.getenv("RAGANYTHING_WORKING_DIR", "./rag_storage"))
DEFAULT_UPLOAD_DIR = Path(os.getenv("RAGANYTHING_UPLOAD_DIR", "./ui_uploads"))
DEFAULT_TEXT_MODEL = os.getenv("RAGANYTHING_TEXT_MODEL", "gpt-4o-mini")
DEFAULT_VISION_MODEL = os.getenv("RAGANYTHING_VISION_MODEL", "gpt-4o")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "RAGANYTHING_EMBEDDING_MODEL", "text-embedding-3-large"
)
DEFAULT_EMBEDDING_DIM = int(os.getenv("RAGANYTHING_EMBEDDING_DIM", "3072"))
DEFAULT_QUERY_MODE = os.getenv("RAGANYTHING_QUERY_MODE", "mix")
DEFAULT_PARSER = os.getenv("RAGANYTHING_PARSER", "mineru")
DEFAULT_PARSE_METHOD = os.getenv("RAGANYTHING_PARSE_METHOD", "auto")
DEFAULT_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
DEFAULT_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))


def _run_async(coro):
    """Execute an async coroutine from synchronous contexts."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    return asyncio.run(coro)


def _format_file_list(file_names: Iterable[str]) -> str:
    files = list(file_names)
    if not files:
        return "No documents processed yet."
    lines = [f"- {name}" for name in files]
    return "Processed documents:\n" + "\n".join(lines)


@dataclass
class AppState:
    """Holds mutable state for the UI."""

    rag: Optional[RAGAnything] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    working_dir: Path = field(default_factory=lambda: DEFAULT_WORKING_DIR)
    upload_dir: Path = field(default_factory=lambda: DEFAULT_UPLOAD_DIR)
    parser: str = DEFAULT_PARSER
    parse_method: str = DEFAULT_PARSE_METHOD
    processed_files: List[str] = field(default_factory=list)

    def ensure_directories(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        api_key: str,
        base_url: Optional[str],
        text_model: str,
        vision_model: str,
        embedding_model: str,
        embedding_dim: int,
        working_dir: Optional[str],
        parser: str,
        parse_method: str,
    ) -> str:
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError(
                "An OpenAI-compatible API key is required. "
                "Provide it in the UI or via the OPENAI_API_KEY environment variable."
            )

        inferred_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or ""
        ).strip()
        self.base_url = inferred_base_url or None

        self.text_model = text_model.strip() or DEFAULT_TEXT_MODEL
        self.vision_model = vision_model.strip()
        self.embedding_model = embedding_model.strip() or DEFAULT_EMBEDDING_MODEL
        self.embedding_dim = int(embedding_dim)
        self.parser = parser or DEFAULT_PARSER
        self.parse_method = parse_method or DEFAULT_PARSE_METHOD

        chosen_working_dir = Path(working_dir.strip()) if working_dir else DEFAULT_WORKING_DIR
        self.working_dir = chosen_working_dir.resolve()
        self.upload_dir = DEFAULT_UPLOAD_DIR.resolve()

        self.ensure_directories()

        config = RAGAnythingConfig(
            working_dir=str(self.working_dir),
            parser=self.parser,
            parse_method=self.parse_method,
        )

        def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return openai_complete_if_cache(
                self.text_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=None,
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # The LightRAG multimodal pipeline may provide structured messages for VLM models.
            if messages is not None:
                return openai_complete_if_cache(
                    self.vision_model or self.text_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )

            if image_data is not None:
                return openai_complete_if_cache(
                    self.vision_model or self.text_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )

            return llm_model_func(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=self.embedding_model,
                api_key=self.api_key,
                base_url=self.base_url,
            ),
        )

        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func if self.vision_model else None,
            embedding_func=embedding_func,
        )

        init_result = _run_async(self.rag._ensure_lightrag_initialized())
        if not init_result["success"]:
            raise RuntimeError(init_result["error"])

        self.processed_files.clear()
        return (
            "RAG-Anything pipeline initialised. Upload documents and start querying!"
        )

    def process_files(
        self, file_paths: Iterable[str], parse_method: Optional[str] = None
    ) -> List[str]:
        if self.rag is None:
            raise RuntimeError("Initialise the pipeline before processing documents.")

        self.ensure_directories()
        statuses: List[str] = []

        for file_path in file_paths:
            if not file_path:
                continue
            src_path = Path(file_path)
            if not src_path.exists():
                statuses.append(f"⚠️ Skipped missing file: {src_path.name}")
                continue

            destination = self.upload_dir / src_path.name
            shutil.copy(src_path, destination)

            _run_async(
                self.rag.process_document_complete(
                    str(destination),
                    parse_method=parse_method or self.parse_method,
                )
            )
            self.processed_files.append(destination.name)
            statuses.append(f"✅ Processed {destination.name}")

        return statuses

    def query(self, message: str, mode: str, vlm_enhanced: bool) -> str:
        if self.rag is None:
            raise RuntimeError("Initialise the pipeline before querying.")

        response = _run_async(
            self.rag.aquery(message, mode=mode, vlm_enhanced=vlm_enhanced)
        )
        return response


app_state = AppState()
app_state.ensure_directories()


def initialise_ui(
    api_key: str,
    base_url: str,
    text_model: str,
    vision_model: str,
    embedding_model: str,
    embedding_dim: float,
    working_dir: str,
    parser: str,
    parse_method: str,
):
    try:
        message = app_state.initialize(
            api_key=api_key,
            base_url=base_url,
            text_model=text_model,
            vision_model=vision_model,
            embedding_model=embedding_model,
            embedding_dim=int(embedding_dim),
            working_dir=working_dir,
            parser=parser,
            parse_method=parse_method,
        )
        return gr.update(value=f"✅ {message}")
    except Exception as exc:  # noqa: BLE001
        return gr.update(value=f"❌ {exc}")


def ingest_documents(file_paths: List[str], parse_method: str):
    if not file_paths:
        return gr.update(value="⚠️ No files selected."), gr.update(
            value=_format_file_list(app_state.processed_files)
        )
    try:
        statuses = app_state.process_files(file_paths, parse_method=parse_method)
        status_text = "\n".join(statuses)
    except Exception as exc:  # noqa: BLE001
        status_text = f"❌ {exc}"
    return gr.update(value=status_text), gr.update(
        value=_format_file_list(app_state.processed_files)
    )


def chat_with_rag(message: str, history: List[tuple[str, str]], mode: str, vlm: bool):
    del history  # Chat history is handled internally by LightRAG
    try:
        return app_state.query(message, mode=mode, vlm_enhanced=vlm)
    except Exception as exc:  # noqa: BLE001
        return f"❌ {exc}"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="RAG-Anything Web UI") as demo:
        gr.Markdown(
            """
            # 🧠 RAG-Anything Web UI

            1. Configure your OpenAI-compatible endpoint and initialise the pipeline.
            2. Upload documents to build the local RAG knowledge base.
            3. Ask questions in the chat panel.
            """
        )

        with gr.Accordion("1. Pipeline configuration", open=True):
            with gr.Row():
                api_key = gr.Textbox(
                    label="API Key",
                    value=os.getenv("OPENAI_API_KEY", ""),
                    type="password",
                    placeholder="sk-...",
                )
                base_url = gr.Textbox(
                    label="Base URL", value=os.getenv("OPENAI_BASE_URL", ""), placeholder="https://api.openai.com/v1"
                )
            with gr.Row():
                text_model = gr.Textbox(
                    label="Text LLM model", value=DEFAULT_TEXT_MODEL
                )
                vision_model = gr.Textbox(
                    label="Vision model (optional)", value=DEFAULT_VISION_MODEL
                )
            with gr.Row():
                embedding_model = gr.Textbox(
                    label="Embedding model", value=DEFAULT_EMBEDDING_MODEL
                )
                embedding_dim = gr.Number(
                    label="Embedding dimension", value=DEFAULT_EMBEDDING_DIM, precision=0
                )
            with gr.Row():
                working_dir = gr.Textbox(
                    label="Working directory", value=str(DEFAULT_WORKING_DIR)
                )
                parser = gr.Dropdown(
                    label="Parser",
                    choices=["mineru", "docling"],
                    value=DEFAULT_PARSER,
                )
                parse_method = gr.Dropdown(
                    label="Parse method",
                    choices=["auto", "ocr", "txt"],
                    value=DEFAULT_PARSE_METHOD,
                )

            init_status = gr.Markdown("Status: waiting for initialisation…")
            init_button = gr.Button("Initialise pipeline", variant="primary")
            init_button.click(
                initialise_ui,
                inputs=[
                    api_key,
                    base_url,
                    text_model,
                    vision_model,
                    embedding_model,
                    embedding_dim,
                    working_dir,
                    parser,
                    parse_method,
                ],
                outputs=init_status,
            )

        with gr.Accordion("2. Document ingestion", open=True):
            with gr.Row():
                file_input = gr.File(
                    label="Upload documents",
                    file_count="multiple",
                    type="filepath",
                )
                parse_override = gr.Dropdown(
                    label="Parse method override",
                    choices=["auto", "ocr", "txt"],
                    value=DEFAULT_PARSE_METHOD,
                )
            ingest_status = gr.Markdown("Upload documents after initialising the pipeline.")
            processed_view = gr.Markdown(_format_file_list(app_state.processed_files))
            ingest_button = gr.Button("Process documents", variant="secondary")
            ingest_button.click(
                ingest_documents,
                inputs=[file_input, parse_override],
                outputs=[ingest_status, processed_view],
            )

        gr.Markdown("---")
        gr.Markdown("## 3. Ask questions about your knowledge base")

        chat = gr.ChatInterface(
            fn=chat_with_rag,
            additional_inputs=[
                gr.Dropdown(
                    label="Query mode",
                    choices=["mix", "local", "global", "hybrid", "naive", "bypass"],
                    value=DEFAULT_QUERY_MODE,
                ),
                gr.Checkbox(label="Enable VLM enhanced query", value=True),
            ],
            title="RAG-Anything Assistant",
            description="Ask questions once your documents are processed.",
        )
        chat.queue()

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(server_name=DEFAULT_SERVER_NAME, server_port=DEFAULT_SERVER_PORT)


if __name__ == "__main__":
    main()
