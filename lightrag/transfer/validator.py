"""
Transfer Validation Module

This module provides validation functionality to ensure knowledge graph
transfers are completed successfully and maintain data integrity.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from ..lightrag import LightRAG
from ..utils import logger


@dataclass
class ValidationResult:
    """Result of transfer validation"""
    success: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class ValidationConfig:
    """Configuration for transfer validation"""
    check_node_count: bool = True
    check_edge_count: bool = True
    check_embeddings: bool = True
    check_query_functionality: bool = True
    sample_queries: List[str] = None
    performance_threshold_seconds: float = 30.0
    similarity_threshold: float = 0.1


class TransferValidator:
    """
    Validates knowledge graph transfers to ensure data integrity and functionality.
    
    This class provides comprehensive validation including:
    - Structural integrity (node/edge counts)
    - Embedding consistency
    - Query functionality
    - Performance benchmarks
    - Data quality checks
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default test queries for functionality validation
        self.default_test_queries = [
            "What are the main topics in this knowledge base?",
            "How are the key concepts related?",
            "Summarize the most important information.",
            "What entities are mentioned most frequently?",
            "Describe the relationships between major concepts."
        ]
    
    async def verify_transfer(
        self,
        source_metadata: Any,
        target_rag: LightRAG,
        config: Optional[ValidationConfig] = None
    ) -> ValidationResult:
        """
        Comprehensive validation of knowledge graph transfer.
        
        Args:
            source_metadata: Metadata from source transfer
            target_rag: Target LightRAG instance
            config: Validation configuration
            
        Returns:
            ValidationResult with detailed validation results
        """
        config = config or ValidationConfig()
        errors = []
        warnings = []
        statistics = {}
        performance_metrics = {}
        
        try:
            self.logger.info("Starting transfer validation...")
            
            # 1. Validate structural integrity
            if config.check_node_count or config.check_edge_count:
                struct_result = await self._validate_structure(
                    source_metadata, target_rag, config
                )
                errors.extend(struct_result["errors"])
                warnings.extend(struct_result["warnings"])
                statistics.update(struct_result["statistics"])
            
            # 2. Validate embeddings
            if config.check_embeddings:
                embed_result = await self._validate_embeddings(target_rag)
                errors.extend(embed_result["errors"])
                warnings.extend(embed_result["warnings"])
                statistics.update(embed_result["statistics"])
            
            # 3. Validate query functionality
            if config.check_query_functionality:
                query_result = await self._validate_query_functionality(
                    target_rag, config
                )
                errors.extend(query_result["errors"])
                warnings.extend(query_result["warnings"])
                performance_metrics.update(query_result["performance"])
            
            # 4. Validate data quality
            quality_result = await self._validate_data_quality(target_rag)
            warnings.extend(quality_result["warnings"])
            statistics.update(quality_result["statistics"])
            
            success = len(errors) == 0
            
            self.logger.info(f"Validation completed: {'SUCCESS' if success else 'FAILED'}")
            self.logger.info(f"Errors: {len(errors)}, Warnings: {len(warnings)}")
            
            return ValidationResult(
                success=success,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed with exception: {str(e)}")
            return ValidationResult(
                success=False,
                errors=[f"Validation exception: {str(e)}"],
                warnings=[],
                statistics={},
                performance_metrics={}
            )
    
    async def _validate_structure(
        self,
        source_metadata: Any,
        target_rag: LightRAG,
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Validate structural integrity of transferred graph"""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            # Get target graph statistics
            target_stats = await self._get_target_statistics(target_rag)
            
            # Compare node counts
            if config.check_node_count:
                source_nodes = getattr(source_metadata, 'node_count', 0)
                target_nodes = target_stats.get('node_count', 0)
                
                statistics['source_nodes'] = source_nodes
                statistics['target_nodes'] = target_nodes
                
                if source_nodes != target_nodes:
                    if abs(source_nodes - target_nodes) / max(source_nodes, 1) > 0.1:
                        errors.append(
                            f"Significant node count mismatch: {source_nodes} -> {target_nodes}"
                        )
                    else:
                        warnings.append(
                            f"Minor node count difference: {source_nodes} -> {target_nodes}"
                        )
                else:
                    self.logger.info(f"Node count validated: {target_nodes}")
            
            # Compare edge counts
            if config.check_edge_count:
                source_edges = getattr(source_metadata, 'edge_count', 0)
                target_edges = target_stats.get('edge_count', 0)
                
                statistics['source_edges'] = source_edges
                statistics['target_edges'] = target_edges
                
                if source_edges != target_edges:
                    if abs(source_edges - target_edges) / max(source_edges, 1) > 0.1:
                        errors.append(
                            f"Significant edge count mismatch: {source_edges} -> {target_edges}"
                        )
                    else:
                        warnings.append(
                            f"Minor edge count difference: {source_edges} -> {target_edges}"
                        )
                else:
                    self.logger.info(f"Edge count validated: {target_edges}")
            
        except Exception as e:
            errors.append(f"Structure validation error: {str(e)}")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "statistics": statistics
        }
    
    async def _validate_embeddings(self, target_rag: LightRAG) -> Dict[str, Any]:
        """Validate embedding consistency and quality"""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            # Check if vector databases exist and are accessible
            vector_stats = await self._check_vector_databases(target_rag)
            statistics.update(vector_stats)
            
            # Validate embedding dimensions
            embedding_info = await self._validate_embedding_dimensions(target_rag)
            if embedding_info.get('inconsistent_dimensions'):
                errors.append("Inconsistent embedding dimensions detected")
            
            statistics.update(embedding_info)
            
            # Check for empty embeddings
            empty_embeddings = embedding_info.get('empty_embeddings', 0)
            if empty_embeddings > 0:
                warnings.append(f"Found {empty_embeddings} empty embeddings")
            
        except Exception as e:
            errors.append(f"Embedding validation error: {str(e)}")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "statistics": statistics
        }
    
    async def _validate_query_functionality(
        self,
        target_rag: LightRAG,
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Validate that query functionality works correctly"""
        errors = []
        warnings = []
        performance = {}
        
        try:
            test_queries = config.sample_queries or self.default_test_queries
            
            for i, query in enumerate(test_queries[:3]):  # Test first 3 queries
                try:
                    # Test different query modes
                    for mode in ["naive", "local", "global"]:
                        start_time = asyncio.get_event_loop().time()
                        
                        result = target_rag.query(query, param={"mode": mode})
                        
                        end_time = asyncio.get_event_loop().time()
                        query_time = end_time - start_time
                        
                        performance[f"query_{i}_{mode}_time"] = query_time
                        
                        # Check if result is reasonable
                        if not result or len(result.strip()) < 10:
                            warnings.append(f"Short/empty result for query {i} in {mode} mode")
                        
                        # Check performance threshold
                        if query_time > config.performance_threshold_seconds:
                            warnings.append(
                                f"Slow query performance: {query_time:.2f}s for {mode} mode"
                            )
                        
                        self.logger.debug(f"Query {i} ({mode}): {query_time:.2f}s")
                
                except Exception as e:
                    errors.append(f"Query {i} failed: {str(e)}")
            
        except Exception as e:
            errors.append(f"Query functionality validation error: {str(e)}")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "performance": performance
        }
    
    async def _validate_data_quality(self, target_rag: LightRAG) -> Dict[str, Any]:
        """Validate overall data quality"""
        warnings = []
        statistics = {}
        
        try:
            # Check for orphaned nodes (nodes without edges)
            orphaned_nodes = await self._count_orphaned_nodes(target_rag)
            if orphaned_nodes > 0:
                warnings.append(f"Found {orphaned_nodes} orphaned nodes")
            statistics['orphaned_nodes'] = orphaned_nodes
            
            # Check for duplicate entities
            duplicate_entities = await self._count_duplicate_entities(target_rag)
            if duplicate_entities > 0:
                warnings.append(f"Found {duplicate_entities} potential duplicate entities")
            statistics['duplicate_entities'] = duplicate_entities
            
            # Check entity description quality
            quality_stats = await self._analyze_entity_descriptions(target_rag)
            statistics.update(quality_stats)
            
            if quality_stats.get('empty_descriptions', 0) > 0:
                warnings.append(f"Found {quality_stats['empty_descriptions']} entities with empty descriptions")
            
        except Exception as e:
            warnings.append(f"Data quality validation error: {str(e)}")
        
        return {
            "warnings": warnings,
            "statistics": statistics
        }
    
    async def _get_target_statistics(self, target_rag: LightRAG) -> Dict[str, int]:
        """Get statistics from target RAG instance using existing LightRAG methods"""
        try:
            # Use existing LightRAG method to get knowledge graph statistics
            kg = await target_rag.get_knowledge_graph("*", max_depth=1, max_nodes=10000)

            # Get document count using existing storage
            doc_count = 0
            if hasattr(target_rag, 'full_docs'):
                try:
                    all_docs = await target_rag.full_docs.get_all()
                    doc_count = len(all_docs) if all_docs else 0
                except Exception:
                    doc_count = 0

            return {
                "node_count": len(kg.nodes) if kg and hasattr(kg, 'nodes') else 0,
                "edge_count": len(kg.edges) if kg and hasattr(kg, 'edges') else 0,
                "document_count": doc_count
            }
        except Exception as e:
            self.logger.warning(f"Could not get target statistics: {str(e)}")
            return {"node_count": 0, "edge_count": 0, "document_count": 0}
    
    async def _check_vector_databases(self, target_rag: LightRAG) -> Dict[str, Any]:
        """Check vector database accessibility using existing LightRAG interfaces"""
        stats = {}

        try:
            # Check entities vector database using existing interface
            if hasattr(target_rag, 'entities_vdb'):
                try:
                    # Try to get a sample to verify accessibility
                    sample_result = await target_rag.entities_vdb.query("test", top_k=1)
                    stats['entities_vdb_accessible'] = True
                    stats['entities_vdb_sample_count'] = len(sample_result) if sample_result else 0
                except Exception:
                    stats['entities_vdb_accessible'] = False

            # Check relationships vector database
            if hasattr(target_rag, 'relationships_vdb'):
                try:
                    sample_result = await target_rag.relationships_vdb.query("test", top_k=1)
                    stats['relationships_vdb_accessible'] = True
                    stats['relationships_vdb_sample_count'] = len(sample_result) if sample_result else 0
                except Exception:
                    stats['relationships_vdb_accessible'] = False

            # Check chunks vector database
            if hasattr(target_rag, 'chunks_vdb'):
                try:
                    sample_result = await target_rag.chunks_vdb.query("test", top_k=1)
                    stats['chunks_vdb_accessible'] = True
                    stats['chunks_vdb_sample_count'] = len(sample_result) if sample_result else 0
                except Exception:
                    stats['chunks_vdb_accessible'] = False

        except Exception as e:
            stats['vector_db_error'] = str(e)

        return stats
    
    async def _validate_embedding_dimensions(self, target_rag: LightRAG) -> Dict[str, Any]:
        """Validate embedding dimensions consistency using existing LightRAG interfaces"""
        info = {
            'inconsistent_dimensions': False,
            'empty_embeddings': 0,
            'dimension_stats': {}
        }

        try:
            # Get expected embedding dimension from target RAG configuration
            expected_dim = None
            if hasattr(target_rag, 'embedding_func') and target_rag.embedding_func:
                expected_dim = getattr(target_rag.embedding_func, 'embedding_dim', None)

            # Check entities vector database
            if hasattr(target_rag, 'entities_vdb'):
                try:
                    # Try to get a sample entity to check embedding dimension
                    sample_entities = await target_rag.entities_vdb.query("sample", top_k=1)
                    if sample_entities:
                        # Assuming the vector database returns results with embeddings
                        entity_dim = len(sample_entities[0].get('embedding', [])) if sample_entities[0].get('embedding') else 0
                        info['dimension_stats']['entities'] = entity_dim
                        if expected_dim and entity_dim != expected_dim:
                            info['inconsistent_dimensions'] = True
                    else:
                        info['dimension_stats']['entities'] = 0
                        info['empty_embeddings'] += 1
                except Exception:
                    info['dimension_stats']['entities'] = 'error_checking'

            # Check relationships vector database
            if hasattr(target_rag, 'relationships_vdb'):
                try:
                    sample_relations = await target_rag.relationships_vdb.query("sample", top_k=1)
                    if sample_relations:
                        relation_dim = len(sample_relations[0].get('embedding', [])) if sample_relations[0].get('embedding') else 0
                        info['dimension_stats']['relationships'] = relation_dim
                        if expected_dim and relation_dim != expected_dim:
                            info['inconsistent_dimensions'] = True
                    else:
                        info['dimension_stats']['relationships'] = 0
                        info['empty_embeddings'] += 1
                except Exception:
                    info['dimension_stats']['relationships'] = 'error_checking'

            # Check chunks vector database
            if hasattr(target_rag, 'chunks_vdb'):
                try:
                    sample_chunks = await target_rag.chunks_vdb.query("sample", top_k=1)
                    if sample_chunks:
                        chunk_dim = len(sample_chunks[0].get('embedding', [])) if sample_chunks[0].get('embedding') else 0
                        info['dimension_stats']['chunks'] = chunk_dim
                        if expected_dim and chunk_dim != expected_dim:
                            info['inconsistent_dimensions'] = True
                    else:
                        info['dimension_stats']['chunks'] = 0
                        info['empty_embeddings'] += 1
                except Exception:
                    info['dimension_stats']['chunks'] = 'error_checking'

            # Add expected dimension for reference
            info['expected_dimension'] = expected_dim

        except Exception as e:
            info['validation_error'] = str(e)

        return info
    
    async def _count_orphaned_nodes(self, target_rag: LightRAG) -> int:
        """Count nodes without any edges using existing graph storage interface"""
        try:
            # Use existing graph storage methods
            if hasattr(target_rag, 'chunk_entity_relation_graph'):
                # Get all nodes and check their connections
                kg = await target_rag.get_knowledge_graph("*", max_depth=1, max_nodes=10000)
                if kg and hasattr(kg, 'nodes') and hasattr(kg, 'edges'):
                    # Count nodes that don't appear in any edge
                    connected_nodes = set()
                    for edge in kg.edges:
                        if hasattr(edge, 'source') and hasattr(edge, 'target'):
                            connected_nodes.add(edge.source)
                            connected_nodes.add(edge.target)

                    total_nodes = len(kg.nodes)
                    orphaned_count = total_nodes - len(connected_nodes)
                    return max(0, orphaned_count)
            return 0
        except Exception as e:
            self.logger.debug(f"Error counting orphaned nodes: {str(e)}")
            return -1  # Error indicator

    async def _count_duplicate_entities(self, target_rag: LightRAG) -> int:
        """Count potential duplicate entities using basic name similarity"""
        try:
            # Get all entities from the knowledge graph
            kg = await target_rag.get_knowledge_graph("*", max_depth=1, max_nodes=10000)
            if not kg or not hasattr(kg, 'nodes'):
                return 0

            # Extract entity names
            entity_names = []
            for node in kg.nodes:
                name = getattr(node, 'entity_name', '') or getattr(node, 'name', '') or str(node)
                if name:
                    entity_names.append(name.lower().strip())

            if not entity_names:
                return 0

            # Simple duplicate detection based on exact name matches
            from collections import Counter
            name_counts = Counter(entity_names)
            duplicates = sum(count - 1 for count in name_counts.values() if count > 1)

            # Basic similarity check for near-duplicates (optional)
            potential_duplicates = 0
            unique_names = list(name_counts.keys())

            for i, name1 in enumerate(unique_names):
                for name2 in unique_names[i+1:]:
                    # Simple similarity check: if names are very similar
                    if len(name1) > 3 and len(name2) > 3:
                        # Check if one name is contained in another or they share significant overlap
                        if (name1 in name2 or name2 in name1 or
                            len(set(name1.split()) & set(name2.split())) > 0):
                            potential_duplicates += 1

            return duplicates + min(potential_duplicates, 10)  # Cap potential duplicates

        except Exception as e:
            self.logger.debug(f"Error counting duplicate entities: {str(e)}")
            return -1  # Error indicator

    async def _analyze_entity_descriptions(self, target_rag: LightRAG) -> Dict[str, int]:
        """Analyze quality of entity descriptions using existing storage"""
        try:
            # Use existing graph storage to analyze entity descriptions
            if hasattr(target_rag, 'chunk_entity_relation_graph'):
                kg = await target_rag.get_knowledge_graph("*", max_depth=1, max_nodes=10000)
                if kg and hasattr(kg, 'nodes'):
                    total_entities = len(kg.nodes)
                    empty_descriptions = 0
                    short_descriptions = 0
                    total_length = 0

                    for node in kg.nodes:
                        description = getattr(node, 'description', '') or ''
                        if not description.strip():
                            empty_descriptions += 1
                        elif len(description.strip()) < 10:
                            short_descriptions += 1
                        total_length += len(description)

                    avg_length = total_length // total_entities if total_entities > 0 else 0

                    return {
                        'total_entities': total_entities,
                        'empty_descriptions': empty_descriptions,
                        'short_descriptions': short_descriptions,
                        'avg_description_length': avg_length
                    }

            return {
                'total_entities': 0,
                'empty_descriptions': 0,
                'short_descriptions': 0,
                'avg_description_length': 0
            }
        except Exception as e:
            self.logger.debug(f"Error analyzing entity descriptions: {str(e)}")
            return {'analysis_error': True}
