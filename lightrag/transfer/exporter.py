"""
Knowledge Graph Export Functionality

This module handles exporting knowledge graph data from a source LightRAG instance
for transfer to another server.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from ..lightrag import LightRAG
from ..utils import logger


class KnowledgeGraphExporter:
    """
    Handles exporting knowledge graph data from LightRAG instances.
    
    Supports exporting:
    - Graph structure (nodes, edges, relationships)
    - Vector embeddings
    - LLM response cache
    - Original documents
    - Metadata and configuration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_structure(self, rag_instance: LightRAG, output_path: Path) -> bool:
        """
        Export knowledge graph structure without embeddings.

        Reuses LightRAG's existing export_data functionality to avoid code duplication.

        Args:
            rag_instance: Source LightRAG instance
            output_path: Directory to export structure data

        Returns:
            True if export successful
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)

            # Use existing export_data method - DRY principle
            self.logger.info("Exporting knowledge graph structure using existing LightRAG export...")

            # Export to Excel format (includes all sheets: Entities, Relations, Relationships)
            excel_path = output_path / "knowledge_graph.xlsx"
            rag_instance.export_data(
                str(excel_path),
                file_format="excel",
                include_vector_data=False  # Structure only, no embeddings
            )

            # Export as CSV for compatibility and easier processing
            csv_path = output_path / "knowledge_graph.csv"
            rag_instance.export_data(
                str(csv_path),
                file_format="csv",
                include_vector_data=False
            )

            # Export raw storage files for direct transfer compatibility
            self._export_raw_storage_files(rag_instance, output_path)

            self.logger.info(f"Structure export completed: {excel_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export structure: {str(e)}")
            return False
    
    def export_embeddings(self, rag_instance: LightRAG, output_path: Path) -> bool:
        """
        Export vector embeddings separately.

        Reuses LightRAG's existing export functionality with vector data included.

        Args:
            rag_instance: Source LightRAG instance
            output_path: Directory to export embeddings

        Returns:
            True if export successful
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)

            # Use existing export_data method with vector data - DRY principle
            self.logger.info("Exporting embeddings using existing LightRAG export...")

            # Export embeddings in Excel format
            embeddings_excel_path = output_path / "embeddings.xlsx"
            rag_instance.export_data(
                str(embeddings_excel_path),
                file_format="excel",
                include_vector_data=True  # Include embeddings
            )

            # Export raw vector database files for direct compatibility
            self._export_vector_databases(rag_instance, output_path)

            self.logger.info(f"Embeddings export completed: {embeddings_excel_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export embeddings: {str(e)}")
            return False
    
    def export_llm_cache(self, rag_instance: LightRAG, output_path: Path) -> bool:
        """
        Export LLM response cache.
        
        Args:
            rag_instance: Source LightRAG instance
            output_path: Directory to export cache
            
        Returns:
            True if export successful
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Copy LLM cache file
            cache_source = Path(rag_instance.working_dir) / "kv_store_llm_response_cache.json"
            if cache_source.exists():
                cache_dest = output_path / "llm_response_cache.json"
                shutil.copy2(cache_source, cache_dest)
                self.logger.info(f"Exported LLM cache: {cache_source} -> {cache_dest}")
            else:
                self.logger.warning("LLM cache file not found")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export LLM cache: {str(e)}")
            return False
    
    def export_documents(
        self,
        rag_instance: LightRAG,
        output_path: Path,
        additional_paths: Optional[List[str]] = None
    ) -> bool:
        """
        Export original documents and text chunks.
        
        Args:
            rag_instance: Source LightRAG instance
            output_path: Directory to export documents
            additional_paths: Additional document paths to include
            
        Returns:
            True if export successful
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export stored documents
            docs_source = Path(rag_instance.working_dir) / "kv_store_full_docs.json"
            if docs_source.exists():
                docs_dest = output_path / "full_documents.json"
                shutil.copy2(docs_source, docs_dest)
                self.logger.info(f"Exported documents: {docs_source} -> {docs_dest}")
            
            # Export text chunks
            chunks_source = Path(rag_instance.working_dir) / "kv_store_text_chunks.json"
            if chunks_source.exists():
                chunks_dest = output_path / "text_chunks.json"
                shutil.copy2(chunks_source, chunks_dest)
                self.logger.info(f"Exported chunks: {chunks_source} -> {chunks_dest}")
            
            # Export additional document paths if provided
            if additional_paths:
                self._export_additional_documents(additional_paths, output_path)
            
            # Create document manifest
            self._create_document_manifest(rag_instance, output_path, additional_paths)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export documents: {str(e)}")
            return False
    
    def _export_raw_storage_files(self, rag_instance: LightRAG, output_path: Path):
        """Export raw storage files for direct transfer - reuses LightRAG storage structure"""
        from ..constants import NameSpace

        raw_dir = output_path / "raw_storage"
        raw_dir.mkdir(exist_ok=True)

        # Use LightRAG's namespace constants to get correct file names
        storage_files = {
            # Graph storage files
            f"{NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION}.graphml": "graph_chunk_entity_relation.graphml",
            # KV storage files
            f"{NameSpace.KV_STORE_FULL_DOCS}.json": "kv_store_full_docs.json",
            f"{NameSpace.KV_STORE_TEXT_CHUNKS}.json": "kv_store_text_chunks.json",
            f"{NameSpace.DOC_STATUS}.json": "doc_status.json"
        }

        source_dir = Path(rag_instance.working_dir)

        for source_name, dest_name in storage_files.items():
            source_file = source_dir / dest_name  # Current file naming
            if source_file.exists():
                dest_file = raw_dir / dest_name
                shutil.copy2(source_file, dest_file)
                self.logger.debug(f"Copied storage file: {dest_name}")

    def _export_vector_databases(self, rag_instance: LightRAG, output_path: Path):
        """Export vector database files - reuses LightRAG namespace structure"""
        from ..constants import NameSpace

        vector_dir = output_path / "vector_dbs"
        vector_dir.mkdir(exist_ok=True)

        # Use LightRAG's namespace constants for vector databases
        vector_files = {
            NameSpace.VECTOR_STORE_ENTITIES: "vdb_entities.json",
            NameSpace.VECTOR_STORE_RELATIONSHIPS: "vdb_relationships.json",
            NameSpace.VECTOR_STORE_CHUNKS: "vdb_chunks.json"
        }

        source_dir = Path(rag_instance.working_dir)

        for namespace, filename in vector_files.items():
            source_file = source_dir / filename
            if source_file.exists():
                dest_file = vector_dir / filename
                shutil.copy2(source_file, dest_file)
                self.logger.debug(f"Copied vector DB: {filename}")
    
    def _export_additional_documents(self, document_paths: List[str], output_path: Path):
        """Export additional document files with PDF processing support"""
        from ..pdf_processor import is_pdf_processing_available

        additional_dir = output_path / "additional_documents"
        additional_dir.mkdir(exist_ok=True)

        pdf_available = is_pdf_processing_available()
        supported_extensions = {'.txt', '.md', '.docx', '.doc'}
        if pdf_available:
            supported_extensions.add('.pdf')

        exported_count = 0
        skipped_count = 0

        for doc_path in document_paths:
            source_path = Path(doc_path)
            if source_path.exists():
                if source_path.is_file():
                    # Check if file type is supported for processing
                    if source_path.suffix.lower() in supported_extensions:
                        dest_path = additional_dir / source_path.name
                        shutil.copy2(source_path, dest_path)
                        exported_count += 1
                        self.logger.info(f"Exported additional document: {source_path}")
                    elif source_path.suffix.lower() == '.pdf' and not pdf_available:
                        self.logger.warning(f"PDF file skipped (no PDF processor available): {source_path}")
                        skipped_count += 1
                    else:
                        # Copy other files anyway, but warn about potential processing issues
                        dest_path = additional_dir / source_path.name
                        shutil.copy2(source_path, dest_path)
                        exported_count += 1
                        self.logger.warning(f"Exported unsupported file type: {source_path}")

                elif source_path.is_dir():
                    dest_path = additional_dir / source_path.name
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    exported_count += 1
                    self.logger.info(f"Exported additional directory: {source_path}")
            else:
                self.logger.warning(f"Document path does not exist: {source_path}")
                skipped_count += 1

        if exported_count > 0 or skipped_count > 0:
            self.logger.info(f"Additional documents export: {exported_count} exported, {skipped_count} skipped")
    
    def _create_document_manifest(
        self,
        rag_instance: LightRAG,
        output_path: Path,
        additional_paths: Optional[List[str]] = None
    ):
        """Create manifest of exported documents"""
        manifest = {
            "export_timestamp": str(Path().cwd()),
            "source_working_dir": rag_instance.working_dir,
            "exported_files": [],
            "additional_paths": additional_paths or []
        }
        
        # List all exported files
        for file_path in output_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(output_path)
                manifest["exported_files"].append({
                    "path": str(relative_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        
        manifest_path = output_path / "document_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Created document manifest: {manifest_path}")
    
    def export_configuration(self, rag_instance: LightRAG, output_path: Path) -> Dict[str, Any]:
        """
        Export LightRAG configuration using existing dataclass structure.

        Reuses LightRAG's existing configuration structure instead of duplicating field extraction.

        Args:
            rag_instance: Source LightRAG instance
            output_path: Directory to export configuration

        Returns:
            Configuration dictionary
        """
        from dataclasses import asdict

        # Use existing LightRAG configuration structure - DRY principle
        try:
            # Get configuration using existing asdict method (same as used in LightRAG.__post_init__)
            config = asdict(rag_instance)

            # Remove non-serializable items
            serializable_config = {}
            for key, value in config.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_config[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable items like functions
                    serializable_config[key] = str(type(value).__name__)

            # Save configuration
            config_path = output_path / "source_configuration.json"
            with open(config_path, 'w') as f:
                json.dump(serializable_config, f, indent=2)

            self.logger.info(f"Exported configuration: {config_path}")
            return serializable_config

        except Exception as e:
            self.logger.warning(f"Could not export full configuration: {str(e)}")
            # Fallback to basic configuration
            basic_config = {
                "working_dir": getattr(rag_instance, 'working_dir', None),
                "export_error": str(e)
            }

            config_path = output_path / "source_configuration.json"
            with open(config_path, 'w') as f:
                json.dump(basic_config, f, indent=2)

            return basic_config
