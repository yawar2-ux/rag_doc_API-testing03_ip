"""
RAG Service: Core component for document retrieval and processing.
"""

import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
import re
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from pathlib import Path

class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None 
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.document_metadata = {}
        self.raw_documents = {}
        self.document_texts = []
        self.bm25 = None
        self.cross_encoder = None
        
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("Cross-encoder model loaded successfully for re-ranking.")
        except Exception as e:
            print(f"Warning: Could not load cross-encoder for re-ranking: {str(e)}")
            self.cross_encoder = None
        
        self.extracted_content_dir = Path("temp/extracted_content")
        self.extracted_content_dir.mkdir(parents=True, exist_ok=True)
        
    def reset_state(self):
        """
        Reset the RAG service to its initial state.
        """
        self.vectorstore = None
        self.document_metadata = {}
        self.raw_documents = {}
        self.document_texts = []
        self.bm25 = None
        print("RAGService state has been reset.")

    def process_file(self, file_path: str, advanced_extraction: bool = False, perform_ocr: bool = False) -> List[str]:
        """Process any supported file type (PDF or text)."""
        if file_path.lower().endswith('.pdf'):
            return self.process_pdf(file_path, advanced_extraction, perform_ocr)
        elif file_path.lower().endswith('.txt'):
            return self.process_text(file_path)
        else:
            return [f"Unsupported file format: {os.path.basename(file_path)}"]
    
    def process_pdf(self, file_path: str, advanced_extraction: bool = False, perform_ocr: bool = False) -> List[str]:
        try:
            filename = os.path.basename(file_path)
            
            if advanced_extraction:
                try:
                    from .docling import convert_pdf_to_markdown # Corrected relative import
                    print(f"Attempting advanced extraction with docling for {filename}.")
                    if perform_ocr:
                        print(f"Note: OCR for images within PDF is currently not supported with docling extraction path.")
                    full_markdown = convert_pdf_to_markdown(file_path)
                    return self.process_markdown(full_markdown, filename)
                except Exception as docling_error:
                    print(f"Docling extraction failed: {str(docling_error)}. Falling back to PyPDF.")
                    # Fall through to PyPDF if advanced_extraction is True but fails
        
            # Standard extraction using PyPDF
            from .pypdf import extract_pdf_content # Corrected relative import
            documents, page_count = extract_pdf_content(file_path, self.document_metadata, self.raw_documents)
            
            if not documents:
                return [f"Warning: No content was extracted from {filename}"]

            # Initialize OCR counters for the current PDF
            total_images_ocr_attempted_in_pdf = 0
            successful_ocr_images_in_pdf = 0

            # Perform OCR on images if requested
            if perform_ocr:
                pdf_doc = None  # Initialize pdf_doc to None
                try:
                    import fitz  # PyMuPDF
                    from .ImageOCR import get_image_description # Corrected relative import
                    import tempfile 

                    print(f"Performing OCR on images in {filename}...")
                    pdf_doc = fitz.open(file_path)
                    
                    for page_index, doc_page in enumerate(documents):
                        ocr_texts = [] 

                        if page_index < len(pdf_doc):
                            fitz_page = pdf_doc.load_page(page_index)
                            image_list = fitz_page.get_images(full=True)
                            
                            if image_list:
                                for img_index, img_info in enumerate(image_list):
                                    xref = img_info[0]
                                    base_image = pdf_doc.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    image_ext = base_image["ext"]
                                    
                                    # Save image to temporary file
                                    temp_ocr_image_path = None # Initialize for finally block
                                    try:
                                        # Using NamedTemporaryFile as in the previous version
                                        # Suffix includes page and image index for better uniqueness if needed, though not strictly necessary
                                        temp_file_for_ocr = tempfile.NamedTemporaryFile(delete=False, suffix=f"_p{page_index}_i{img_index}.{image_ext}")
                                        temp_ocr_image_path = temp_file_for_ocr.name
                                        temp_file_for_ocr.write(image_bytes)
                                        temp_file_for_ocr.close()
                                        
                                        print(f"  OCR processing image {img_index+1} on page {page_index+1}")
                                        total_images_ocr_attempted_in_pdf += 1 # Increment attempt counter
                                        image_text = get_image_description(temp_ocr_image_path)
                                        
                                        if not image_text.startswith("Error:"):
                                            successful_ocr_images_in_pdf += 1 # Increment success counter
                                            ocr_texts.append(image_text)
                                        # else: failed_ocr_images_in_pdf is implicitly (total_images_ocr_attempted_in_pdf - successful_ocr_images_in_pdf)
                                    finally:
                                        if temp_ocr_image_path and os.path.exists(temp_ocr_image_path):
                                            os.remove(temp_ocr_image_path)
                            
                            # Add OCR text to document content
                            if ocr_texts: # This check is now safe
                                ocr_content = "\n\n[OCR CONTENT FROM PAGE IMAGES]:\n" + "\n---\n".join(ocr_texts)
                                doc_page.page_content += ocr_content
                                
                                # Update raw_documents as well
                                key = f"{filename}_page_{doc_page.metadata['page']}"
                                if key in self.raw_documents:
                                    self.raw_documents[key] += ocr_content
                
                    print(f"OCR processing completed for {filename}") # Moved inside try block
                
                except ModuleNotFoundError as mnfe:
                    # Check if the error is specifically about the 'frontend' module,
                    # which is a known issue with corrupted PyMuPDF installations.
                    if mnfe.name == 'frontend':
                        detailed_error_msg = (
                            "PyMuPDF (fitz) failed to load a required internal component ('frontend'). "
                            "This often indicates an issue with the PyMuPDF installation. "
                            "Consider reinstalling it: 'pip uninstall PyMuPDF' then 'pip install PyMuPDF'."
                        )
                        print(f"OCR Initialization Error for {filename}: {detailed_error_msg}")
                        return [f"Error during OCR of {filename}: {detailed_error_msg}"]
                    else:
                        # Handle other ModuleNotFoundErrors (e.g., if Services.ImageOCR was missing)
                        error_msg = f"A required module ('{mnfe.name}') for OCR is missing."
                        print(f"OCR module import error for {filename}: {mnfe}. {error_msg}")
                        return [f"Error during OCR of {filename}: {error_msg}"]
                except Exception as e: # Catch other exceptions during OCR processing
                    error_msg = f"An unexpected error occurred during OCR: {str(e)}"
                    print(f"Error during OCR processing for {filename}: {str(e)}")
                    return [f"Error during OCR of {filename}: {error_msg}"]
                finally:
                    if pdf_doc: # Check if pdf_doc was successfully opened
                        pdf_doc.close() # Moved to finally block to ensure it's closed
            
            chunks = self.text_splitter.split_documents(documents)
            
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks, 
                    embedding=self.embeddings
                )
            else:
                self.vectorstore.add_documents(chunks)
            
            # Construct the final status message
            final_status_message = f"Processed {filename}: {len(chunks)} chunks from {page_count} pages"

            if perform_ocr:
                if total_images_ocr_attempted_in_pdf > 0:
                    failed_ocr_count = total_images_ocr_attempted_in_pdf - successful_ocr_images_in_pdf
                    ocr_details = f". OCR: {successful_ocr_images_in_pdf}/{total_images_ocr_attempted_in_pdf} images succeeded"
                    if failed_ocr_count > 0:
                        ocr_details += f" ({failed_ocr_count} failed)."
                    else:
                        ocr_details += "."
                    final_status_message += ocr_details
                else:
                    final_status_message += ". OCR: No images found to process."
            
            return [final_status_message]
            
        except Exception as e:
            return [f"Error processing PDF {os.path.basename(file_path)}: {str(e)}"]
    
    def process_text(self, file_path: str) -> List[str]:
        """Process text files using TextLoader."""
        try:
            
            
            filename = os.path.basename(file_path)
            loader = TextLoader(file_path)
            documents = loader.load()
            
            for i, doc in enumerate(documents):
                doc.metadata['page'] = i + 1
                doc.metadata['source'] = filename
                doc.metadata['filename'] = filename
                
                key = f"{filename}_chunk_{i+1}" # Changed from page to chunk for clarity
                self.document_metadata[key] = {
                    'filename': filename,
                    'page_number': i + 1 # Or 'chunk_number' if more appropriate
                }
                self.raw_documents[key] = doc.page_content
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Create or update vector store
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks, 
                    embedding=self.embeddings
                )
            else:
                self.vectorstore.add_documents(chunks)
                
            return [f"Processed text file {filename}: {len(chunks)} chunks created"]
            
        except Exception as e:
            return [f"Error processing text file: {str(e)}"]
    
    def process_markdown(self, markdown_content: str, source_filename: str) -> List[str]:
        """Process markdown content, typically from docling."""
        try:
            from langchain.document_loaders import TextLoader # Keep this specific import here
            
            # Sanitize filename
            if '_' in source_filename:
                parts = source_filename.split('_', 1)
                # Check if the first part looks like a UUID (common pattern from unique naming)
                try:
                    uuid_candidate = parts[0]
                    if len(uuid_candidate) == 36 and uuid_candidate.count('-') == 4: # Basic UUID check
                         clean_filename = parts[1]
                    else:
                         clean_filename = source_filename
                except: # Handle cases where split might not work as expected
                    clean_filename = source_filename

            else:
                clean_filename = source_filename
                
            base_name = os.path.splitext(clean_filename)[0]
            safe_base_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in base_name)
            
            md_path = os.path.join(self.extracted_content_dir, f"{safe_base_name}.md")
            with open(md_path, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)
            
            txt_path = os.path.join(self.extracted_content_dir, f"{safe_base_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(markdown_content) # Storing raw markdown as .txt for TextLoader
            
            print(f"Markdown content saved to {md_path} and {txt_path} for processing.")
            
            if not os.path.exists(txt_path):
                return [f"Error: Text file {txt_path} for markdown processing was not created successfully"]
                
            try:
                loader = TextLoader(txt_path, encoding='utf-8')
                documents = loader.load()
            except Exception as loader_error:
                return [f"Error loading text file for markdown: {str(loader_error)}. File path: {txt_path}"]
            
            for i, doc in enumerate(documents):
                section_num = i + 1
                doc.metadata['page'] = section_num # 'page' might represent section in markdown context
                doc.metadata['source'] = source_filename # Original uploaded name
                doc.metadata['filename'] = clean_filename # Processed filename
                doc.metadata['markdown_path'] = md_path
                doc.metadata['text_path'] = txt_path
                
                key = f"{clean_filename}_md_section_{section_num}"
                self.document_metadata[key] = {
                    'filename': clean_filename,
                    'page_number': section_num,
                    'source_type': 'markdown',
                    'markdown_path': md_path,
                    'text_path': txt_path
                }
                self.raw_documents[key] = doc.page_content
            
            markdown_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, # Larger chunk for markdown
                chunk_overlap=200,
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "] # Added H4
            )
            chunks = markdown_splitter.split_documents(documents)
            
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks, 
                    embedding=self.embeddings
                )
            else:
                self.vectorstore.add_documents(chunks)
                
            return [f"Processed markdown for {clean_filename}: {len(chunks)} chunks created. Content from {source_filename}."]
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            return [f"Error processing markdown from {source_filename}: {str(e)}\n{trace}"]
    
    def _build_bm25_index(self):
        """Build or rebuild the BM25 index from document chunks."""
        if not self.raw_documents:
            self.bm25 = None
            return
            
        # Extract texts and create tokenized corpus
        texts = list(self.raw_documents.values())
        self.document_texts = texts
        
        # Tokenize the documents
        tokenized_corpus = [doc.lower().split() for doc in texts]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Built BM25 index with {len(self.document_texts)} documents.") # Changed from texts to self.document_texts
    
    def retrieve_relevant_docs(self, query: str, k: int = 5, hybrid_alpha: float = 0.7, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Retrieve documents relevant to the query using hybrid search (semantic + BM25)."""
        if not self.vectorstore:
            return []
        
        if not self.bm25 and self.raw_documents: # Build BM25 if not present and there's data
            self._build_bm25_index()
        
        k_candidates = max(k * 3, 15) 
        
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k_candidates)
        semantic_results = []
        for doc, score in docs_with_scores:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 0)
            
            semantic_results.append({
                'content': doc.page_content,
                'metadata': {
                    'filename': source,
                    'page_number': page,
                    'semantic_score': float(score)
                },
                'doc_obj': doc
            })
        
        # Step 2: Perform BM25 search if available
        if self.bm25 and self.document_texts:
            # Tokenize the query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k_candidates document indices by BM25 score
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_candidates]
            
            # Collect BM25 results
            bm25_results = []
            for idx in top_bm25_indices:
                # Find document metadata by content
                content = self.document_texts[idx]
                doc_key = None
                for key, text in self.raw_documents.items():
                    if text == content:
                        doc_key = key
                        break
                
                if doc_key and doc_key in self.document_metadata:
                    metadata = self.document_metadata[doc_key]
                    bm25_results.append({
                        'content': content,
                        'metadata': {
                            'filename': metadata.get('filename', 'Unknown'),
                            'page_number': metadata.get('page_number', 0),
                            'bm25_score': float(bm25_scores[idx])
                        }
                    })
            
            # Step 3: Combine semantic and BM25 results with hybrid scoring
            combined_results = {}
            
            # Add semantic results with normalized scores (lower distance score is better, convert to similarity)
            # Filter out results with perfect distance (0.0) if they are too many, or handle max_semantic_score being 0
            valid_semantic_scores = [r['metadata']['semantic_score'] for r in semantic_results if r['metadata']['semantic_score'] > 0]
            max_semantic_score = max(valid_semantic_scores) if valid_semantic_scores else 1.0 
            # min_semantic_score = min([r['metadata']['semantic_score'] for r in semantic_results]) if semantic_results else 0.0


            for result in semantic_results:
                content = result['content']
                # Normalize score: 1 - (score / max_score) for distance scores to make higher better
                # If score is 0 (perfect match), it becomes 1. If score is max_score, it becomes 0.
                # Chroma's L2 distance is >= 0.
                score = result['metadata']['semantic_score']
                normalized_score = 1.0 - (score / max_semantic_score) if max_semantic_score > 0 else (0.0 if score > 0 else 1.0)
                
                if content not in combined_results:
                    combined_results[content] = {
                        'content': content,
                        'metadata': result['metadata'].copy(),
                        'hybrid_score': hybrid_alpha * normalized_score,
                        'doc_obj': result.get('doc_obj') # Keep original doc object if needed later
                    }
                # else: # If content is already there, could average or take max, but usually content is unique per chunk
                    # combined_results[content]['hybrid_score'] = max(combined_results[content]['hybrid_score'], hybrid_alpha * normalized_score)


            # Add BM25 results with normalized scores (higher BM25 score is better)
            valid_bm25_scores = [r['metadata']['bm25_score'] for r in bm25_results if r['metadata']['bm25_score'] > 0]
            max_bm25_score = max(valid_bm25_scores) if valid_bm25_scores else 1.0

            for result in bm25_results:
                content = result['content']
                normalized_score = result['metadata']['bm25_score'] / max_bm25_score if max_bm25_score > 0 else 0.0
                if content in combined_results:
                    combined_results[content]['metadata']['bm25_score'] = result['metadata']['bm25_score']
                    combined_results[content]['hybrid_score'] += (1 - hybrid_alpha) * normalized_score
                else:
                    combined_results[content] = {
                        'content': content,
                        'metadata': result['metadata'].copy(),
                        'hybrid_score': (1 - hybrid_alpha) * normalized_score
                        # 'doc_obj' would be missing here if only from BM25 and not semantic
                    }
            
            results = list(combined_results.values())
            results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            if use_reranking and self.cross_encoder and results: # Ensure results exist before reranking
                # Rerank top N candidates from hybrid results
                candidates_to_rerank = results[:k_candidates] 
                
                pairs = [(query, doc['content']) for doc in candidates_to_rerank]
                if not pairs: # if no candidates, skip reranking
                    pass
                else:
                    cross_scores = self.cross_encoder.predict(pairs)
                    
                    for i, score in enumerate(cross_scores):
                        candidates_to_rerank[i]['metadata']['cross_encoder_score'] = float(score)
                        candidates_to_rerank[i]['final_score'] = float(score) # Use cross_encoder_score as the final score
                    
                    candidates_to_rerank.sort(key=lambda x: x['final_score'], reverse=True)
                    # The final list of results will be these reranked candidates
                    # Potentially, other non-reranked results could be appended if needed, but typically not.
                    results = candidates_to_rerank 
        else:
            # Fallback to semantic results if BM25 is not available
            # Sort by semantic score (lower is better for distance, so reverse=False or use 1-score)
            semantic_results.sort(key=lambda x: x['metadata']['semantic_score'])
            results = semantic_results # These are already in the desired format
            # If reranking is enabled and cross-encoder exists, rerank pure semantic results
            if use_reranking and self.cross_encoder and results:
                candidates_to_rerank = results[:k_candidates]
                pairs = [(query, doc['content']) for doc in candidates_to_rerank]
                if pairs:
                    cross_scores = self.cross_encoder.predict(pairs)
                    for i, score in enumerate(cross_scores):
                        candidates_to_rerank[i]['metadata']['cross_encoder_score'] = float(score)
                        candidates_to_rerank[i]['final_score'] = float(score)
                    candidates_to_rerank.sort(key=lambda x: x['final_score'], reverse=True)
                    results = candidates_to_rerank
        
        final_results = []
        for i, result in enumerate(results[:k]): # Take top k results
            # Determine the score to report: final_score (reranked), hybrid_score, or semantic_score
            score_to_report = result.get('final_score', result.get('hybrid_score', result['metadata'].get('semantic_score', 0)))
            if 'semantic_score' in result['metadata'] and not result.get('final_score') and not result.get('hybrid_score'):
                # If only semantic score is available and it's a distance, might want to invert/normalize for display
                # For now, just pass it as is.
                pass

            final_dict = {
                'content': result['content'],
                'metadata': {
                    'filename': result['metadata'].get('filename', 'Unknown'),
                    'page_number': result['metadata'].get('page_number', 0), # Ensure page_number is present
                    'score': score_to_report
                }
            }
            # Add other scores if present for debugging
            if 'semantic_score' in result['metadata']:
                final_dict['metadata']['debug_semantic_score'] = result['metadata']['semantic_score']
            if 'bm25_score' in result['metadata']:
                final_dict['metadata']['debug_bm25_score'] = result['metadata']['bm25_score']
            if 'cross_encoder_score' in result['metadata']:
                final_dict['metadata']['debug_cross_encoder_score'] = result['metadata']['cross_encoder_score']
            if 'hybrid_score' in result:
                 final_dict['metadata']['debug_hybrid_score'] = result['hybrid_score']


            final_results.append(final_dict)
        
        return final_results
    
    def summarize_context(self, documents: List[Dict[str, Any]], max_tokens: int = 6000) -> str:
        """Combine retrieved documents into a single context string."""
        context_parts = []
        estimated_tokens = 0
        
        for doc in documents:
            page_ref = f"[Page {doc['metadata']['page_number']}]"
            content = doc['content']
            estimated_chunk_tokens = len(content) / 4
            
            if estimated_tokens + estimated_chunk_tokens > max_tokens and context_parts:
                break
                
            context_parts.append(f"{page_ref} {content}")
            estimated_tokens += estimated_chunk_tokens
        
        return "\n\n".join(context_parts)
    
    def extract_source_references(self, text: str, context: str) -> List[Dict[str, Any]]:
        """Extract source references from the response text and context."""
        page_pattern = r'\[Page (\d+)\]' # Matches [Page X]
        
        # Find all unique page numbers mentioned in the context
        context_pages = set(re.findall(page_pattern, context))
        
        # Find all unique page numbers mentioned in the LLM's answer
        answer_pages = set(re.findall(page_pattern, text))
        
        # Consider pages relevant if they are in the context AND mentioned in the answer,
        # or if they are simply in the answer (LLM might hallucinate page numbers not in context).
        # A safer approach is to only list sources that were part of the provided context.
        relevant_pages_in_answer = answer_pages.intersection(context_pages)
        
        # If no pages from context are in answer, but answer mentions pages,
        # it might be a general statement or hallucination.
        # For now, we list any page mentioned in the answer that has a known source.
        
        pages_to_find_sources_for = context_pages.union(answer_pages) # Show all potential sources

        sources = []
        found_pages_meta = set()

        for page_str in sorted(list(pages_to_find_sources_for), key=int): # Sort pages numerically
            # Search for this page number in our document metadata
            # This assumes page numbers are unique across all documents, or filename helps disambiguate
            # The current self.document_metadata might have keys like "filename_page_X" or "filename_md_section_X"
            
            # We need a more robust way to link page numbers back to their original filenames
            # from the `documents` list passed to `summarize_context`.
            # The `documents` in `summarize_context` already have 'filename' and 'page_number'.
            
            filename_for_page = "Unknown" # Default
            
            # Try to find the filename associated with this page number from self.document_metadata
            # This part can be tricky if page numbers are not globally unique without filenames.
            for key, meta_info in self.document_metadata.items():
                if str(meta_info.get('page_number')) == page_str:
                    # Prioritize if this page was actually in the LLM's answer
                    if page_str in answer_pages:
                        filename_for_page = meta_info.get('filename', "Document")
                        source_entry = (filename_for_page, page_str)
                        if source_entry not in found_pages_meta:
                            sources.append({
                                "page_number": page_str,
                                "filename": filename_for_page,
                                "in_answer": True 
                            })
                            found_pages_meta.add(source_entry)
                        break # Found one, assume it's good enough for now
            
            # If not found in answer pages' direct metadata lookup, check context pages
            if (filename_for_page, page_str) not in found_pages_meta and page_str in context_pages:
                 for key, meta_info in self.document_metadata.items():
                    if str(meta_info.get('page_number')) == page_str:
                        filename_for_page = meta_info.get('filename', "Document")
                        source_entry = (filename_for_page, page_str)
                        if source_entry not in found_pages_meta: # Avoid duplicates
                            sources.append({
                                "page_number": page_str,
                                "filename": filename_for_page,
                                "in_answer": False # Present in context, but not explicitly cited by LLM
                            })
                            found_pages_meta.add(source_entry)
                        break
        
        # Sort sources: those in answer first, then by filename, then by page number
        sources.sort(key=lambda x: (not x['in_answer'], x['filename'], int(x['page_number'])))
        
        return sources