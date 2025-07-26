"""
Knowledge Graph Import Functionality

This module handles importing knowledge graph data into a target LightRAG instance
from a transfer package created by the exporter.
"""

import os
import json
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import pandas as pd

from ..lightrag import LightRAG
from ..utils import logger


class KnowledgeGraphImporter:
    """
    Handles importing knowledge graph data into LightRAG instances.
    
    Supports importing:
    - Graph structure (nodes, edges, relationships)
    - Vector embeddings (if compatible)
    - LLM response cache
    - Original documents
    - Regenerating embeddings with new models
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def import_structure(self, target_rag: LightRAG, structure_path: Path) -> bool:
        """
        Import knowledge graph structure into target instance.
        
        Args:
            target_rag: Target LightRAG instance
            structure_path: Path to exported structure data
            
        Returns:
            True if import successful
        """
        try:
            # Method 1: Import using custom KG format (recommended)
            await self._import_via_custom_kg(target_rag, structure_path)
            
            # Method 2: Direct file copy (fallback)
            # await self._import_via_direct_copy(target_rag, structure_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import structure: {str(e)}")
            return False
    
    async def import_embeddings(self, target_rag: LightRAG, embeddings_path: Path) -> bool:
        """
        Import vector embeddings into target instance.
        
        Args:
            target_rag: Target LightRAG instance
            embeddings_path: Path to exported embeddings
            
        Returns:
            True if import successful
        """
        try:
            # Copy vector database files directly
            vector_db_dir = embeddings_path / "vector_dbs"
            if vector_db_dir.exists():
                target_dir = Path(target_rag.working_dir)
                
                vector_files = [
                    "vdb_entities.json",
                    "vdb_relationships.json",
                    "vdb_chunks.json"
                ]
                
                for filename in vector_files:
                    source_file = vector_db_dir / filename
                    if source_file.exists():
                        dest_file = target_dir / filename
                        shutil.copy2(source_file, dest_file)
                        self.logger.info(f"Imported vector DB: {filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import embeddings: {str(e)}")
            return False
    
    async def import_llm_cache(self, target_rag: LightRAG, cache_path: Path) -> bool:
        """
        Import LLM response cache into target instance.
        
        Args:
            target_rag: Target LightRAG instance
            cache_path: Path to exported cache
            
        Returns:
            True if import successful
        """
        try:
            cache_file = cache_path / "llm_response_cache.json"
            if cache_file.exists():
                target_cache = Path(target_rag.working_dir) / "kv_store_llm_response_cache.json"
                
                # Merge with existing cache if present
                if target_cache.exists():
                    await self._merge_llm_caches(cache_file, target_cache)
                else:
                    shutil.copy2(cache_file, target_cache)
                
                self.logger.info(f"Imported LLM cache: {cache_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import LLM cache: {str(e)}")
            return False
    
    async def import_documents(self, target_rag: LightRAG, documents_path: Path) -> bool:
        """
        Import documents and re-process them with target LLM.
        
        Args:
            target_rag: Target LightRAG instance
            documents_path: Path to exported documents
            
        Returns:
            True if import successful
        """
        try:
            # Import stored documents
            full_docs_file = documents_path / "full_documents.json"
            if full_docs_file.exists():
                target_docs = Path(target_rag.working_dir) / "kv_store_full_docs.json"
                shutil.copy2(full_docs_file, target_docs)
                self.logger.info("Imported full documents")
            
            # Import text chunks
            chunks_file = documents_path / "text_chunks.json"
            if chunks_file.exists():
                target_chunks = Path(target_rag.working_dir) / "kv_store_text_chunks.json"
                shutil.copy2(chunks_file, target_chunks)
                self.logger.info("Imported text chunks")
            
            # Process additional documents if present
            additional_docs_dir = documents_path / "additional_documents"
            if additional_docs_dir.exists():
                await self._process_additional_documents(target_rag, additional_docs_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import documents: {str(e)}")
            return False
    
    async def regenerate_embeddings(
        self,
        target_rag: LightRAG,
        structure_path: Path,
        batch_size: int = 100
    ) -> bool:
        """
        Regenerate all embeddings using target embedding model.

        This method reuses LightRAG's existing embedding generation workflow
        by re-processing documents through the normal insertion pipeline.

        Args:
            target_rag: Target LightRAG instance
            structure_path: Path to structure data
            batch_size: Batch size for processing

        Returns:
            True if regeneration successful
        """
        try:
            self.logger.info("Starting embedding regeneration using existing LightRAG workflow...")

            # Method 1: Re-process original documents (most reliable)
            documents_path = structure_path.parent / "documents"
            if documents_path.exists():
                await self._reprocess_documents(target_rag, documents_path)

            # Method 2: Re-embed existing entities and relationships using existing utilities
            await self._regenerate_entity_embeddings(target_rag, structure_path)
            await self._regenerate_relationship_embeddings(target_rag, structure_path)

            # Trigger final storage update
            await target_rag._insert_done()

            self.logger.info("Embedding regeneration completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to regenerate embeddings: {str(e)}")
            return False
    
    async def _import_via_custom_kg(self, target_rag: LightRAG, structure_path: Path):
        """Import using LightRAG's custom KG format"""
        excel_file = structure_path / "knowledge_graph.xlsx"
        if not excel_file.exists():
            raise FileNotFoundError("Knowledge graph Excel file not found")
        
        # Read exported data
        excel_data = pd.ExcelFile(str(excel_file))
        
        # Load entities
        entities_df = pd.read_excel(excel_data, sheet_name='Entities')
        entities_list = []
        for _, row in entities_df.iterrows():
            entity = {
                "entity_name": row.get("entity_name", ""),
                "entity_type": row.get("entity_type", "UNKNOWN"),
                "description": row.get("description", ""),
                "source_id": row.get("source_id", "")
            }
            # Add optional fields if present
            for field in ["file_path", "created_at"]:
                if field in row and pd.notna(row[field]):
                    entity[field] = row[field]
            entities_list.append(entity)
        
        # Load relationships
        relations_df = pd.read_excel(excel_data, sheet_name='Relations')
        relationships_list = []
        for _, row in relations_df.iterrows():
            relationship = {
                "src_id": row.get("src_id", ""),
                "tgt_id": row.get("tgt_id", ""),
                "description": row.get("description", ""),
                "keywords": row.get("keywords", ""),
                "weight": float(row.get("weight", 1.0)),
                "source_id": row.get("source_id", "")
            }
            # Add optional fields if present
            for field in ["file_path", "created_at"]:
                if field in row and pd.notna(row[field]):
                    relationship[field] = row[field]
            relationships_list.append(relationship)
        
        # Create custom KG structure
        custom_kg = {
            "entities": entities_list,
            "relationships": relationships_list,
            "chunks": []  # Will be loaded separately or regenerated
        }
        
        # Import into target RAG
        await target_rag.ainsert_custom_kg(custom_kg)
        self.logger.info(f"Imported {len(entities_list)} entities and {len(relationships_list)} relationships")
    
    async def _import_via_direct_copy(self, target_rag: LightRAG, structure_path: Path):
        """Import by directly copying storage files"""
        raw_storage_dir = structure_path / "raw_storage"
        if not raw_storage_dir.exists():
            raise FileNotFoundError("Raw storage directory not found")
        
        target_dir = Path(target_rag.working_dir)
        
        # Files to copy
        files_to_copy = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "doc_status.json"
        ]
        
        for filename in files_to_copy:
            source_file = raw_storage_dir / filename
            if source_file.exists():
                dest_file = target_dir / filename
                shutil.copy2(source_file, dest_file)
                self.logger.info(f"Copied {filename}")
    
    async def _merge_llm_caches(self, source_cache: Path, target_cache: Path):
        """Merge LLM caches, preserving existing entries"""
        # Load source cache
        with open(source_cache, 'r') as f:
            source_data = json.load(f)
        
        # Load target cache if exists
        target_data = {}
        if target_cache.exists():
            with open(target_cache, 'r') as f:
                target_data = json.load(f)
        
        # Merge caches (source takes precedence for conflicts)
        merged_data = {**target_data, **source_data}
        
        # Save merged cache
        with open(target_cache, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        self.logger.info(f"Merged LLM caches: {len(source_data)} + {len(target_data)} = {len(merged_data)}")
    
    async def _process_additional_documents(self, target_rag: LightRAG, docs_dir: Path):
        """Process additional documents using existing LightRAG document processing with PDF support"""
        from ..utils import clean_text
        from ..pdf_processor import PDFProcessorManager, is_pdf_processing_available

        # Initialize PDF processor manager
        pdf_manager = PDFProcessorManager()

        # Define supported file extensions
        supported_extensions = {'.txt', '.md', '.docx', '.doc'}

        # Add PDF support if processor is available
        pdf_available = is_pdf_processing_available()
        if pdf_available:
            supported_extensions.add('.pdf')
            self.logger.info("PDF processing is available for document transfer")
        else:
            self.logger.warning("PDF processing not available - PDF files will be skipped")

        processed_count = 0
        skipped_count = 0

        for doc_path in docs_dir.rglob("*"):
            if doc_path.is_file() and doc_path.suffix.lower() in supported_extensions:
                try:
                    # Extract content based on file type
                    if doc_path.suffix.lower() == '.pdf':
                        # Use PDF processor wrapper with proper error handling
                        try:
                            content = pdf_manager.extract_text(doc_path)
                            if not content.strip():
                                self.logger.warning(f"No text extracted from PDF: {doc_path.name}")
                                skipped_count += 1
                                continue

                            # Get PDF metadata for better processing
                            metadata = pdf_manager.get_metadata(doc_path)
                            self.logger.debug(f"PDF metadata for {doc_path.name}: {metadata.get('page_count', 'unknown')} pages")

                        except Exception as e:
                            self.logger.warning(f"Failed to extract text from PDF {doc_path.name}: {str(e)}")
                            skipped_count += 1
                            continue

                    elif doc_path.suffix.lower() in {'.txt', '.md'}:
                        # Handle text files with proper encoding detection
                        try:
                            with open(doc_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            # Try alternative encodings
                            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                                try:
                                    with open(doc_path, 'r', encoding=encoding) as f:
                                        content = f.read()
                                    self.logger.debug(f"Successfully read {doc_path.name} with {encoding} encoding")
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                self.logger.warning(f"Could not decode text file: {doc_path.name}")
                                skipped_count += 1
                                continue

                    else:
                        # Handle other document types (placeholder for future expansion)
                        self.logger.warning(f"Unsupported document type: {doc_path.suffix} for {doc_path.name}")
                        skipped_count += 1
                        continue

                    # Clean and process the content
                    if content.strip():
                        # Use existing LightRAG text cleaning
                        cleaned_content = clean_text(content)

                        if cleaned_content.strip():
                            # Use existing ainsert method with file path for proper citation
                            await target_rag.ainsert(
                                cleaned_content,
                                file_paths=str(doc_path)
                            )
                            processed_count += 1
                            self.logger.info(f"Processed additional document: {doc_path.name}")
                        else:
                            self.logger.warning(f"Document became empty after cleaning: {doc_path.name}")
                            skipped_count += 1
                    else:
                        self.logger.warning(f"Empty document: {doc_path.name}")
                        skipped_count += 1

                except Exception as e:
                    self.logger.warning(f"Failed to process {doc_path}: {str(e)}")
                    skipped_count += 1

        self.logger.info(f"Additional document processing completed: {processed_count} processed, {skipped_count} skipped")
    
    async def _extract_content_for_embedding(self, structure_path: Path) -> List[Dict[str, Any]]:
        """Extract text content that needs re-embedding"""
        content_list = []
        
        # Extract from documents
        docs_file = structure_path / "raw_storage" / "kv_store_full_docs.json"
        if docs_file.exists():
            with open(docs_file, 'r') as f:
                docs_data = json.load(f)
                for doc_id, doc_info in docs_data.items():
                    content_list.append({
                        "type": "document",
                        "id": doc_id,
                        "content": doc_info.get("content", "")
                    })
        
        # Extract from chunks
        chunks_file = structure_path / "raw_storage" / "kv_store_text_chunks.json"
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
                for chunk_id, chunk_info in chunks_data.items():
                    content_list.append({
                        "type": "chunk",
                        "id": chunk_id,
                        "content": chunk_info.get("content", "")
                    })
        
        return content_list
    
    async def _reprocess_documents(self, target_rag: LightRAG, documents_path: Path):
        """Re-process original documents to regenerate embeddings with enhanced document handling"""
        from ..pdf_processor import PDFProcessorManager

        # Initialize PDF processor for potential use
        pdf_manager = PDFProcessorManager()

        # Process stored full documents
        full_docs_file = documents_path / "full_documents.json"
        if full_docs_file.exists():
            with open(full_docs_file, 'r') as f:
                full_docs = json.load(f)

            self.logger.info(f"Re-processing {len(full_docs)} stored documents...")
            processed_count = 0

            for doc_id, doc_data in full_docs.items():
                content = doc_data.get("content", "")
                file_path = doc_data.get("file_path", "")

                if content:
                    try:
                        # Use existing ainsert method to regenerate embeddings
                        await target_rag.ainsert(
                            content,
                            ids=doc_id,
                            file_paths=file_path if file_path else None
                        )
                        processed_count += 1
                        self.logger.debug(f"Re-processed document: {doc_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to re-process document {doc_id}: {str(e)}")

            self.logger.info(f"Completed re-processing: {processed_count}/{len(full_docs)} documents")

        # Process additional documents with enhanced handling
        additional_docs_dir = documents_path / "additional_documents"
        if additional_docs_dir.exists():
            self.logger.info("Processing additional documents from transfer package...")
            await self._process_additional_documents(target_rag, additional_docs_dir)

    async def _regenerate_entity_embeddings(self, target_rag: LightRAG, structure_path: Path):
        """Regenerate entity embeddings using existing LightRAG utilities"""
        from ..utils import compute_mdhash_id

        try:
            # Load entities from exported structure
            excel_file = structure_path / "knowledge_graph.xlsx"
            if not excel_file.exists():
                return

            excel_data = pd.ExcelFile(str(excel_file))
            entities_df = pd.read_excel(excel_data, sheet_name='Entities')

            self.logger.info(f"Regenerating embeddings for {len(entities_df)} entities...")

            # Process entities in batches
            batch_size = 50
            for i in range(0, len(entities_df), batch_size):
                batch_df = entities_df.iloc[i:i + batch_size]
                entity_data = {}

                for _, row in batch_df.iterrows():
                    entity_name = row.get("entity_name", "")
                    description = row.get("description", "")
                    entity_type = row.get("entity_type", "UNKNOWN")

                    if entity_name and description:
                        # Create content for embedding (same format as LightRAG uses)
                        content = f"{entity_name}\n{entity_type}\n{description}"
                        entity_id = compute_mdhash_id(entity_name, prefix="ent-")

                        entity_data[entity_id] = {
                            "content": content,
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "description": description,
                            "source_id": row.get("source_id", ""),
                            "file_path": row.get("file_path", "")
                        }

                # Upsert batch to vector database (this will generate embeddings)
                if entity_data:
                    await target_rag.entities_vdb.upsert(entity_data)
                    self.logger.debug(f"Regenerated embeddings for entity batch {i//batch_size + 1}")

        except Exception as e:
            self.logger.warning(f"Failed to regenerate entity embeddings: {str(e)}")

    async def _regenerate_relationship_embeddings(self, target_rag: LightRAG, structure_path: Path):
        """Regenerate relationship embeddings using existing LightRAG utilities"""
        from ..utils import compute_mdhash_id

        try:
            # Load relationships from exported structure
            excel_file = structure_path / "knowledge_graph.xlsx"
            if not excel_file.exists():
                return

            excel_data = pd.ExcelFile(str(excel_file))
            relations_df = pd.read_excel(excel_data, sheet_name='Relations')

            self.logger.info(f"Regenerating embeddings for {len(relations_df)} relationships...")

            # Process relationships in batches
            batch_size = 50
            for i in range(0, len(relations_df), batch_size):
                batch_df = relations_df.iloc[i:i + batch_size]
                relation_data = {}

                for _, row in batch_df.iterrows():
                    src_id = row.get("src_id", "")
                    tgt_id = row.get("tgt_id", "")
                    description = row.get("description", "")
                    keywords = row.get("keywords", "")

                    if src_id and tgt_id:
                        # Create content for embedding (same format as LightRAG uses)
                        content = f"{src_id}\t{tgt_id}\n{keywords}\n{description}"
                        relation_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")

                        relation_data[relation_id] = {
                            "content": content,
                            "src_id": src_id,
                            "tgt_id": tgt_id,
                            "description": description,
                            "keywords": keywords,
                            "weight": float(row.get("weight", 1.0)),
                            "source_id": row.get("source_id", ""),
                            "file_path": row.get("file_path", "")
                        }

                # Upsert batch to vector database (this will generate embeddings)
                if relation_data:
                    await target_rag.relationships_vdb.upsert(relation_data)
                    self.logger.debug(f"Regenerated embeddings for relationship batch {i//batch_size + 1}")

        except Exception as e:
            self.logger.warning(f"Failed to regenerate relationship embeddings: {str(e)}")
