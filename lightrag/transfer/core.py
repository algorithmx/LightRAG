"""
Core Knowledge Graph Transfer Implementation

This module provides the main KnowledgeGraphTransfer class that orchestrates
the entire transfer process between LightRAG instances.
"""

import os
import json
import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from ..lightrag import LightRAG
from ..utils import logger
from .exporter import KnowledgeGraphExporter
from .importer import KnowledgeGraphImporter
from .validator import TransferValidator
from .compatibility import EmbeddingCompatibilityHandler


@dataclass
class TransferConfig:
    """Configuration for knowledge graph transfer"""
    include_embeddings: bool = False
    include_llm_cache: bool = True
    include_documents: bool = True
    compression_enabled: bool = True
    verify_integrity: bool = True
    regenerate_embeddings: bool = True
    batch_size: int = 100
    max_retries: int = 3


@dataclass
class TransferMetadata:
    """Metadata about the transfer package"""
    transfer_id: str
    created_at: str
    source_config: Dict[str, Any]
    target_config: Optional[Dict[str, Any]] = None
    node_count: int = 0
    edge_count: int = 0
    document_count: int = 0
    embedding_model_source: Optional[str] = None
    embedding_model_target: Optional[str] = None
    embedding_dim_source: Optional[int] = None
    embedding_dim_target: Optional[int] = None


class KnowledgeGraphTransfer:
    """
    Main class for transferring knowledge graphs between LightRAG instances.
    
    This class handles the complete transfer workflow:
    1. Export from source instance
    2. Package for transfer
    3. Import to target instance
    4. Verify integrity
    5. Handle embedding compatibility
    """
    
    def __init__(self, source_rag: Optional[LightRAG] = None):
        """
        Initialize transfer manager.
        
        Args:
            source_rag: Source LightRAG instance (for export operations)
        """
        self.source_rag = source_rag
        self.exporter = KnowledgeGraphExporter()
        self.importer = KnowledgeGraphImporter()
        self.validator = TransferValidator()
        self.compatibility_handler = EmbeddingCompatibilityHandler()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def export_for_transfer(
        self,
        output_path: str,
        config: Optional[TransferConfig] = None,
        document_paths: Optional[List[str]] = None
    ) -> str:
        """
        Export knowledge graph for transfer to another server.
        
        Args:
            output_path: Path where transfer package will be created
            config: Transfer configuration
            document_paths: Additional document paths to include
            
        Returns:
            Path to created transfer package
        """
        if not self.source_rag:
            raise ValueError("Source RAG instance required for export")
        
        config = config or TransferConfig()
        
        # Create transfer directory
        transfer_dir = Path(output_path)
        transfer_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate transfer metadata
        metadata = self._generate_transfer_metadata(config)
        
        # Export knowledge graph data
        self.logger.info("Exporting knowledge graph structure...")
        structure_path = transfer_dir / "structure"
        self.exporter.export_structure(self.source_rag, structure_path)
        
        # Export embeddings if requested
        if config.include_embeddings:
            self.logger.info("Exporting embeddings...")
            embeddings_path = transfer_dir / "embeddings"
            self.exporter.export_embeddings(self.source_rag, embeddings_path)
        
        # Export LLM cache if requested
        if config.include_llm_cache:
            self.logger.info("Exporting LLM response cache...")
            cache_path = transfer_dir / "cache"
            self.exporter.export_llm_cache(self.source_rag, cache_path)
        
        # Export documents if requested
        if config.include_documents:
            self.logger.info("Exporting documents...")
            docs_path = transfer_dir / "documents"
            self.exporter.export_documents(
                self.source_rag, docs_path, document_paths
            )
        
        # Save metadata
        metadata_path = transfer_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2)
        
        # Create transfer package
        if config.compression_enabled:
            package_path = self._create_compressed_package(transfer_dir)
            # Clean up uncompressed directory
            shutil.rmtree(transfer_dir)
            return package_path
        
        return str(transfer_dir)
    
    async def import_from_transfer(
        self,
        target_rag: LightRAG,
        transfer_package_path: str,
        config: Optional[TransferConfig] = None
    ) -> bool:
        """
        Import knowledge graph from transfer package to target instance.
        
        Args:
            target_rag: Target LightRAG instance
            transfer_package_path: Path to transfer package
            config: Transfer configuration
            
        Returns:
            True if import successful, False otherwise
        """
        config = config or TransferConfig()
        
        # Extract package if compressed
        if transfer_package_path.endswith(('.tar.gz', '.zip')):
            extract_dir = self._extract_package(transfer_package_path)
        else:
            extract_dir = Path(transfer_package_path)
        
        try:
            # Load metadata
            metadata_path = extract_dir / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError("Transfer metadata not found")
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                metadata = TransferMetadata(**metadata_dict)
            
            # Check embedding compatibility
            target_config = self._get_target_config(target_rag)
            compatibility_result = self.compatibility_handler.check_compatibility(
                metadata, target_config
            )
            
            if not compatibility_result.compatible and config.regenerate_embeddings:
                self.logger.info("Embedding models incompatible, will regenerate embeddings")
            
            # Import structure
            self.logger.info("Importing knowledge graph structure...")
            structure_path = extract_dir / "structure"
            await self.importer.import_structure(target_rag, structure_path)
            
            # Handle embeddings
            if config.include_embeddings and not config.regenerate_embeddings:
                if compatibility_result.compatible:
                    self.logger.info("Importing compatible embeddings...")
                    embeddings_path = extract_dir / "embeddings"
                    await self.importer.import_embeddings(target_rag, embeddings_path)
                else:
                    self.logger.warning("Skipping incompatible embeddings")
            
            # Import LLM cache
            if config.include_llm_cache:
                self.logger.info("Importing LLM cache...")
                cache_path = extract_dir / "cache"
                await self.importer.import_llm_cache(target_rag, cache_path)
            
            # Regenerate embeddings if needed
            if config.regenerate_embeddings:
                self.logger.info("Regenerating embeddings with target model...")
                structure_path = extract_dir / "structure"
                await self.importer.regenerate_embeddings(target_rag, structure_path)
            
            # Verify transfer if requested
            if config.verify_integrity:
                self.logger.info("Verifying transfer integrity...")
                verification_result = await self.validator.verify_transfer(
                    metadata, target_rag
                )
                if not verification_result.success:
                    self.logger.error(f"Transfer verification failed: {verification_result.errors}")
                    return False
            
            self.logger.info("Knowledge graph transfer completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Transfer import failed: {str(e)}")
            return False
        
        finally:
            # Clean up extracted files if they were compressed
            if transfer_package_path.endswith(('.tar.gz', '.zip')):
                shutil.rmtree(extract_dir, ignore_errors=True)
    
    def _generate_transfer_metadata(self, config: TransferConfig) -> TransferMetadata:
        """Generate metadata for the transfer package"""
        transfer_id = f"transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get source configuration
        source_config = {
            "working_dir": self.source_rag.working_dir,
            "graph_storage": self.source_rag.graph_storage,
            "vector_storage": self.source_rag.vector_storage,
            "kv_storage": self.source_rag.kv_storage,
        }
        
        # Get embedding information
        embedding_info = self._get_embedding_info(self.source_rag)
        
        # Count nodes and edges
        graph_stats = self._get_graph_statistics()
        
        return TransferMetadata(
            transfer_id=transfer_id,
            created_at=datetime.now().isoformat(),
            source_config=source_config,
            node_count=graph_stats.get("nodes", 0),
            edge_count=graph_stats.get("edges", 0),
            document_count=graph_stats.get("documents", 0),
            embedding_model_source=embedding_info.get("model"),
            embedding_dim_source=embedding_info.get("dimension")
        )
    
    def _get_embedding_info(self, rag_instance: LightRAG) -> Dict[str, Any]:
        """Extract embedding model information from RAG instance"""
        try:
            if hasattr(rag_instance, 'embedding_func') and rag_instance.embedding_func:
                return {
                    "model": getattr(rag_instance.embedding_func, 'model_name', 'unknown'),
                    "dimension": getattr(rag_instance.embedding_func, 'embedding_dim', None)
                }
        except Exception:
            pass
        return {"model": "unknown", "dimension": None}
    
    def _get_graph_statistics(self) -> Dict[str, int]:
        """Get statistics about the source graph using existing LightRAG functionality"""
        try:
            # Use existing LightRAG method to get graph statistics
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Get knowledge graph using existing method
                kg = loop.run_until_complete(
                    self.source_rag.get_knowledge_graph("*", max_depth=1, max_nodes=10000)
                )

                # Count documents from KV storage
                doc_count = 0
                if hasattr(self.source_rag, 'full_docs'):
                    try:
                        all_docs = loop.run_until_complete(self.source_rag.full_docs.get_all())
                        doc_count = len(all_docs) if all_docs else 0
                    except Exception:
                        doc_count = 0

                return {
                    "nodes": len(kg.nodes) if kg and hasattr(kg, 'nodes') else 0,
                    "edges": len(kg.edges) if kg and hasattr(kg, 'edges') else 0,
                    "documents": doc_count
                }
            finally:
                loop.close()

        except Exception as e:
            self.logger.warning(f"Could not get graph statistics: {str(e)}")
            return {"nodes": 0, "edges": 0, "documents": 0}
    
    def _get_target_config(self, target_rag: LightRAG) -> Dict[str, Any]:
        """Get configuration from target RAG instance"""
        return {
            "working_dir": target_rag.working_dir,
            "graph_storage": target_rag.graph_storage,
            "vector_storage": target_rag.vector_storage,
            "kv_storage": target_rag.kv_storage,
            "embedding_info": self._get_embedding_info(target_rag)
        }
    
    def _create_compressed_package(self, transfer_dir: Path) -> str:
        """Create compressed transfer package using standard library"""
        import shutil

        # Use shutil.make_archive for better cross-platform compatibility
        package_path = shutil.make_archive(
            str(transfer_dir),
            'gztar',  # gzipped tar format
            root_dir=transfer_dir.parent,
            base_dir=transfer_dir.name
        )

        return package_path

    def _extract_package(self, package_path: str) -> Path:
        """Extract compressed transfer package using standard library"""
        import shutil
        import tempfile

        extract_dir = Path(tempfile.mkdtemp(prefix="lightrag_transfer_"))

        # Use shutil.unpack_archive for better format support
        try:
            shutil.unpack_archive(package_path, extract_dir)
        except Exception as e:
            self.logger.error(f"Failed to extract package: {str(e)}")
            raise

        # Return the actual extracted directory (first subdirectory)
        extracted_items = list(extract_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            return extracted_items[0]

        return extract_dir
    
    async def _regenerate_embeddings(self, target_rag: LightRAG, extract_dir: Path):
        """
        Regenerate embeddings using target embedding model.

        This method is now deprecated in favor of using the importer's
        regenerate_embeddings method which properly reuses LightRAG functionality.
        """
        self.logger.warning("_regenerate_embeddings is deprecated, use importer.regenerate_embeddings instead")
        structure_path = extract_dir / "structure"
        return await self.importer.regenerate_embeddings(target_rag, structure_path)
