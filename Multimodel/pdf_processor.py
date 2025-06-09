from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import pdfplumber
import pandas as pd
from PIL import Image
import io
import os
import logging
from Multimodel.summary_generator import get_image_description, get_table_description
from Multimodel.vector_store import EnhancedVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PDFChunk:
    chunk_num: int
    image: Optional[dict] = None
    text_sections: List[str] = None
    tables: List[object] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None

    def __post_init__(self):
        self.text_sections = []
        self.tables = []

def is_within_chunk_bounds(pos_y: float, page_num: int, chunk_start: tuple, chunk_end: tuple) -> bool:
    """Check if a position falls within the chunk bounds"""
    if page_num < chunk_start[0] or page_num > chunk_end[0]:
        return False
    if page_num == chunk_start[0] and pos_y < chunk_start[1]:
        return False
    if page_num == chunk_end[0] and pos_y >= chunk_end[1]:
        return False
    return True

async def save_chunk_content(chunk: PDFChunk, base_dir: Path) -> None:
    """Save all content for a chunk and generate summaries"""
    chunk_dir = base_dir / f'chunk_{chunk.chunk_num}'
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Save image and get description
    if chunk.image:
        try:
            image_path = chunk_dir / 'image.png'
            Image.open(io.BytesIO(chunk.image['stream'].get_data())).save(image_path)
            logger.info(f"Saved image for chunk {chunk.chunk_num}")

            # Generate and save image description asynchronously
            image_description = await get_image_description(image_path)
            (chunk_dir / 'image_description.txt').write_text(image_description, encoding='utf-8')
            logger.info(f"Generated image description for chunk {chunk.chunk_num}")
        except Exception as e:
            logger.error(f"Error processing image in chunk {chunk.chunk_num}: {str(e)}")

    # Save text content
    if chunk.text_sections:
        try:
            (chunk_dir / 'text.txt').write_text('\n\n'.join(chunk.text_sections), encoding='utf-8')
            logger.info(f"Saved text content for chunk {chunk.chunk_num}")
        except Exception as e:
            logger.error(f"Error saving text in chunk {chunk.chunk_num}: {str(e)}")

    # Save tables and get descriptions
    if chunk.tables:
        for i, table in enumerate(chunk.tables, 1):
            try:
                table_path = chunk_dir / f'table_{i}.csv'
                df = pd.DataFrame(table.extract())
                headers = df.iloc[0]
                df = df.iloc[1:].set_axis(headers, axis=1)
                df.replace('', pd.NA).dropna(how='all').to_csv(table_path, index=False)
                logger.info(f"Saved table {i} for chunk {chunk.chunk_num}")

                # Generate and save table description asynchronously
                table_description = await get_table_description(table_path)
                (chunk_dir / f'table_{i}_description.txt').write_text(table_description, encoding='utf-8')
                logger.info(f"Generated description for table {i} in chunk {chunk.chunk_num}")
            except Exception as e:
                logger.error(f"Error processing table {i} in chunk {chunk.chunk_num}: {str(e)}")

def extract_chunks_from_pdf(pdf_path: str, start_chunk_num: int = 1) -> List[PDFChunk]:
    """Extract chunks from a single PDF file"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"Processing PDF: {pdf_path}")
            image_positions = []

            # Collect all images and their positions
            for page_num, page in enumerate(pdf.pages):
                for image in page.images:
                    image_positions.append({
                        'pos': (page_num, image['top']),
                        'image': image
                    })

            image_positions.sort(key=lambda x: (x['pos'][0], x['pos'][1]))
            logger.info(f"Found {len(image_positions)} images in PDF")

            if not image_positions:
                logger.warning(f"No images found in {pdf_path}")
                return []

            chunks = []
            for i, curr_img in enumerate(image_positions):
                chunk = PDFChunk(chunk_num=start_chunk_num + i, image=curr_img['image'])
                chunk_start = curr_img['pos']
                chunk_end = image_positions[i+1]['pos'] if i < len(image_positions)-1 else (len(pdf.pages)-1, float('inf'))

                for page_num in range(chunk_start[0], chunk_end[0] + 1):
                    page = pdf.pages[page_num]
                    page_words = []

                    # Extract tables
                    page_tables = [t for t in page.find_tables()
                                 if is_within_chunk_bounds(t.bbox[1], page_num, chunk_start, chunk_end)]
                    chunk.tables.extend(page_tables)

                    # Extract words
                    words = [w for w in page.extract_words()
                            if is_within_chunk_bounds(w['top'], page_num, chunk_start, chunk_end)]

                    # Filter out words that overlap with tables
                    for word in sorted(words, key=lambda w: w['top']):
                        if not any(word['top'] >= t.bbox[1] and word['bottom'] <= t.bbox[3] and
                                 word['x0'] >= t.bbox[0] and word['x1'] <= t.bbox[2] for t in page_tables):
                            page_words.append(word['text'])

                    if page_words:
                        chunk.text_sections.append(' '.join(page_words))

                chunks.append(chunk)
                logger.info(f"Processed chunk {chunk.chunk_num}")

            return chunks

    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return []

async def process_multiple_pdfs(pdf_paths: List[str], output_dir: Path) -> Dict:
    """Process multiple PDF files and save their chunks"""
    current_chunk_num = 1
    all_chunks = []
    chunks_info = {}

    for pdf_path in pdf_paths:
        try:
            chunks = extract_chunks_from_pdf(pdf_path, current_chunk_num)
            all_chunks.extend(chunks)
            chunks_info[pdf_path] = len(chunks)
            current_chunk_num += len(chunks)
            logger.info(f"Successfully processed {pdf_path}")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            chunks_info[pdf_path] = 0

    # Process chunks concurrently
    for chunk in all_chunks:
        try:
            await save_chunk_content(chunk, output_dir)
        except Exception as e:
            logger.error(f"Error saving chunk {chunk.chunk_num}: {str(e)}")

    # Vectorize and store chunks
    try:
        logger.info("Starting vectorization process")
        vector_store = EnhancedVectorStore()
        vector_store.add_chunks_to_db(output_dir)
        chunks_in_store = vector_store.get_total_chunks()
        logger.info(f"Successfully vectorized {chunks_in_store} chunks")
        chunks_info["vectorized_chunks"] = chunks_in_store
    except Exception as e:
        logger.error(f"Error in vectorization process: {str(e)}")
        chunks_info["vectorization_error"] = str(e)

    return {
        "total_chunks": len(all_chunks),
        "chunks_per_pdf": chunks_info
    }