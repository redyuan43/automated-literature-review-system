from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from chemical_formula_processor import ChemicalFormulaProcessor


class DocAgent:
    def __init__(self,
                 chapters_dir: str = "./chapter_markdowns",
                 outputs_dir: str = "./outputs",
                 embeddings_dir: str = "./embeddings"):
        self.chapters_dir = Path(chapters_dir)
        self.outputs_dir = Path(outputs_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.formula = ChemicalFormulaProcessor()

    # File/section IO
    def list_chapters(self) -> List[str]:
        return [p.name for p in self.chapters_dir.glob('*.md')]

    def get_chapter_path(self, chapter_name: str) -> Path:
        return self.chapters_dir / chapter_name

    def get_section(self, chapter_name: str, section_heading: str) -> Optional[str]:
        p = self.get_chapter_path(chapter_name)
        if not p.exists():
            return None
        text = p.read_text(encoding='utf-8')
        # naive extraction by heading
        marker = f"## {section_heading}"
        idx = text.find(marker)
        if idx < 0:
            return None
        tail = text[idx + len(marker):]
        # end at next heading
        next_idx = tail.find("\n## ")
        return tail[:next_idx].strip() if next_idx >= 0 else tail.strip()

    def set_section(self, chapter_name: str, section_heading: str, content: str) -> None:
        p = self.get_chapter_path(chapter_name)
        text = p.read_text(encoding='utf-8') if p.exists() else f"# {chapter_name}\n\n"
        marker = f"## {section_heading}"
        if marker in text:
            head, sep, tail = text.partition(marker)
            # remove old section body
            next_idx = tail.find("\n## ")
            if next_idx >= 0:
                tail = tail[next_idx:]
            else:
                tail = ""
            new_text = head + marker + "\n\n" + content.strip() + "\n\n" + tail
        else:
            new_text = text.rstrip() + f"\n\n{marker}\n\n{content.strip()}\n"
        p.write_text(new_text, encoding='utf-8')

    # RAG
    def load_vector_store(self) -> FAISS:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.load_local(str(self.embeddings_dir), embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        store = self.load_vector_store()
        docs = store.similarity_search(query, k=top_k)
        return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

    def get_context_for(self, chapter_name: str, section_heading: str, limit_tokens: int = 2000) -> str:
        section = self.get_section(chapter_name, section_heading) or ""
        return section[:limit_tokens]

    # Citations
    def list_references(self, chapter_name: str) -> List[Dict[str, Any]]:
        # placeholder: search for a REFERENCES section
        refs = self.get_section(chapter_name, "REFERENCES") or ""
        return [{"text": line.strip()} for line in refs.splitlines() if line.strip()]

    def validate_citation(self, chapter_name: str, section_heading: str) -> str:
        # placeholder: minimal local analysis summary
        section = self.get_section(chapter_name, section_heading) or ""
        refs = self.list_references(chapter_name)
        return f"共检测到{len(refs)}条参考，章节长度{len(section)}字符。"

    # Formulas
    def protect_formulas(self, text: str) -> str:
        return self.formula.preserve_chemical_content(text)

    def restore_formulas(self, text: str) -> str:
        return self.formula.restore_chemical_content(text)

    # Append citation analysis to the end of chapter
    def append_citation_analysis(self, chapter_name: str, analysis: str):
        p = self.get_chapter_path(chapter_name)
        if not p.exists():
            return
        content = p.read_text(encoding='utf-8')
        content += "\n\n## 引用准确性分析\n\n" + analysis.strip() + "\n"
        p.write_text(content, encoding='utf-8')


