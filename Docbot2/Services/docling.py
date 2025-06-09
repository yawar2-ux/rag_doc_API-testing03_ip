from docling.document_converter import DocumentConverter

def convert_pdf_to_markdown(file_path: str) -> str:
    """Convert PDF to markdown using docling."""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()