"""
Embedding Compatibility Handler

This module handles compatibility checks and transformations between different
embedding models used in source and target LightRAG instances.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from ..utils import logger


@dataclass
class CompatibilityResult:
    """Result of embedding compatibility check"""
    compatible: bool
    reason: str
    source_dim: Optional[int] = None
    target_dim: Optional[int] = None
    transformation_needed: bool = False
    transformation_method: Optional[str] = None


@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model"""
    model_name: str
    dimension: int
    max_token_size: int
    similarity_metric: str = "cosine"
    normalization: bool = True


class EmbeddingCompatibilityHandler:
    """
    Handles compatibility between different embedding models.
    
    This class provides:
    - Compatibility checking between source and target embedding models
    - Dimension transformation methods
    - Embedding space alignment techniques
    - Performance optimization for different embedding characteristics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Known embedding model configurations
        self.known_models = {
            "text-embedding-3-large": EmbeddingModelInfo(
                model_name="text-embedding-3-large",
                dimension=3072,
                max_token_size=8192
            ),
            "text-embedding-3-small": EmbeddingModelInfo(
                model_name="text-embedding-3-small", 
                dimension=1536,
                max_token_size=8192
            ),
            "text-embedding-ada-002": EmbeddingModelInfo(
                model_name="text-embedding-ada-002",
                dimension=1536,
                max_token_size=8191
            ),
            "nomic-embed-text": EmbeddingModelInfo(
                model_name="nomic-embed-text",
                dimension=768,
                max_token_size=2048
            ),
            "all-MiniLM-L6-v2": EmbeddingModelInfo(
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                max_token_size=512
            ),
            "all-mpnet-base-v2": EmbeddingModelInfo(
                model_name="all-mpnet-base-v2",
                dimension=768,
                max_token_size=514
            )
        }
    
    def check_compatibility(
        self,
        source_metadata: Any,
        target_config: Dict[str, Any]
    ) -> CompatibilityResult:
        """
        Check compatibility between source and target embedding models.
        
        Args:
            source_metadata: Transfer metadata from source
            target_config: Target RAG configuration
            
        Returns:
            CompatibilityResult with compatibility status and details
        """
        try:
            # Extract embedding information
            source_model = getattr(source_metadata, 'embedding_model_source', None)
            source_dim = getattr(source_metadata, 'embedding_dim_source', None)
            
            target_embedding_info = target_config.get('embedding_info', {})
            target_model = target_embedding_info.get('model', 'unknown')
            target_dim = target_embedding_info.get('dimension', None)
            
            self.logger.info(f"Checking compatibility: {source_model} ({source_dim}D) -> {target_model} ({target_dim}D)")
            
            # Perfect match - can transfer embeddings directly
            if source_model == target_model and source_dim == target_dim:
                return CompatibilityResult(
                    compatible=True,
                    reason="Identical embedding models and dimensions",
                    source_dim=source_dim,
                    target_dim=target_dim
                )
            
            # Dimension mismatch - requires regeneration for best results
            if source_dim and target_dim and source_dim != target_dim:
                return CompatibilityResult(
                    compatible=False,
                    reason=f"Dimension mismatch: {source_dim}D vs {target_dim}D - regeneration recommended",
                    source_dim=source_dim,
                    target_dim=target_dim,
                    transformation_needed=True,
                    transformation_method="full_regeneration"  # Recommend full regeneration for dimension mismatches
                )
            
            # Different models, same dimension - still recommend regeneration
            if source_dim == target_dim:
                return CompatibilityResult(
                    compatible=False,
                    reason=f"Different embedding models: {source_model} vs {target_model} - regeneration recommended for optimal performance",
                    source_dim=source_dim,
                    target_dim=target_dim,
                    transformation_needed=True,
                    transformation_method="full_regeneration"  # Different models have different embedding spaces
                )
            
            # Unknown compatibility
            return CompatibilityResult(
                compatible=False,
                reason="Unknown compatibility - insufficient information",
                source_dim=source_dim,
                target_dim=target_dim,
                transformation_needed=True,
                transformation_method="full_regeneration"
            )
            
        except Exception as e:
            self.logger.error(f"Compatibility check failed: {str(e)}")
            return CompatibilityResult(
                compatible=False,
                reason=f"Compatibility check error: {str(e)}",
                transformation_needed=True,
                transformation_method="full_regeneration"
            )
    
    def get_model_info(self, model_name: str) -> Optional[EmbeddingModelInfo]:
        """
        Get information about a known embedding model.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            EmbeddingModelInfo if known, None otherwise
        """
        return self.known_models.get(model_name)
    
    def suggest_compatibility_strategy(
        self,
        compatibility_result: CompatibilityResult
    ) -> Dict[str, Any]:
        """
        Suggest the best strategy for handling incompatible embeddings.
        
        Args:
            compatibility_result: Result from compatibility check
            
        Returns:
            Dictionary with strategy recommendations
        """
        if compatibility_result.compatible:
            return {
                "strategy": "direct_transfer",
                "description": "Embeddings are compatible, direct transfer possible",
                "performance_impact": "none",
                "accuracy_impact": "none"
            }
        
        if compatibility_result.transformation_method == "full_regeneration":
            return {
                "strategy": "full_regeneration",
                "description": "Regenerate all embeddings with target model",
                "performance_impact": "high",
                "accuracy_impact": "optimal",
                "estimated_time": "high",
                "recommended": True
            }
        
        if compatibility_result.transformation_method == "dimension_reduction":
            return {
                "strategy": "dimension_reduction",
                "description": f"Reduce dimensions from {compatibility_result.source_dim} to {compatibility_result.target_dim}",
                "performance_impact": "medium",
                "accuracy_impact": "medium",
                "methods": ["pca", "truncation", "learned_projection"],
                "recommended": False,
                "note": "May lose semantic information"
            }
        
        if compatibility_result.transformation_method == "dimension_expansion":
            return {
                "strategy": "dimension_expansion", 
                "description": f"Expand dimensions from {compatibility_result.source_dim} to {compatibility_result.target_dim}",
                "performance_impact": "low",
                "accuracy_impact": "poor",
                "methods": ["zero_padding", "random_projection", "learned_expansion"],
                "recommended": False,
                "note": "Likely to degrade performance significantly"
            }
        
        return {
            "strategy": "full_regeneration",
            "description": "Default to full regeneration for unknown cases",
            "performance_impact": "high",
            "accuracy_impact": "optimal",
            "recommended": True
        }
    
    def transform_embeddings(
        self,
        embeddings: np.ndarray,
        source_dim: int,
        target_dim: int,
        method: str = "pca"
    ) -> np.ndarray:
        """
        Transform embeddings between different dimensions.
        
        Args:
            embeddings: Source embeddings array
            source_dim: Source dimension
            target_dim: Target dimension
            method: Transformation method
            
        Returns:
            Transformed embeddings array
        """
        if source_dim == target_dim:
            return embeddings
        
        if method == "truncation" and source_dim > target_dim:
            return embeddings[:, :target_dim]
        
        if method == "zero_padding" and source_dim < target_dim:
            padding = np.zeros((embeddings.shape[0], target_dim - source_dim))
            return np.concatenate([embeddings, padding], axis=1)
        
        if method == "pca" and source_dim > target_dim:
            return self._apply_pca_reduction(embeddings, target_dim)
        
        if method == "random_projection":
            return self._apply_random_projection(embeddings, target_dim)
        
        raise ValueError(f"Unsupported transformation method: {method}")
    
    def _apply_pca_reduction(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Apply PCA for dimension reduction"""
        try:
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            self.logger.info(f"PCA reduction: {embeddings.shape[1]}D -> {target_dim}D, "
                           f"explained variance: {explained_variance:.3f}")
            
            return reduced_embeddings
            
        except ImportError:
            self.logger.warning("scikit-learn not available, falling back to truncation")
            return embeddings[:, :target_dim]
    
    def _apply_random_projection(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Apply random projection for dimension transformation"""
        try:
            from sklearn.random_projection import GaussianRandomProjection
            
            transformer = GaussianRandomProjection(n_components=target_dim, random_state=42)
            transformed_embeddings = transformer.fit_transform(embeddings)
            
            self.logger.info(f"Random projection: {embeddings.shape[1]}D -> {target_dim}D")
            
            return transformed_embeddings
            
        except ImportError:
            self.logger.warning("scikit-learn not available, using simple projection")
            # Simple random projection without sklearn
            projection_matrix = np.random.normal(0, 1, (embeddings.shape[1], target_dim))
            return np.dot(embeddings, projection_matrix)
    
    def optimize_similarity_threshold(
        self,
        source_model: str,
        target_model: str,
        default_threshold: float = 0.2
    ) -> float:
        """
        Optimize similarity threshold for different embedding models.
        
        Args:
            source_model: Source embedding model name
            target_model: Target embedding model name
            default_threshold: Default similarity threshold
            
        Returns:
            Optimized threshold value
        """
        # Model-specific threshold adjustments
        threshold_adjustments = {
            ("text-embedding-3-large", "nomic-embed-text"): 0.3,
            ("text-embedding-ada-002", "all-MiniLM-L6-v2"): 0.4,
            ("nomic-embed-text", "all-mpnet-base-v2"): 0.25,
        }
        
        # Check for specific model pair
        model_pair = (source_model, target_model)
        if model_pair in threshold_adjustments:
            optimized_threshold = threshold_adjustments[model_pair]
            self.logger.info(f"Using optimized threshold {optimized_threshold} for {model_pair}")
            return optimized_threshold
        
        # Check reverse pair
        reverse_pair = (target_model, source_model)
        if reverse_pair in threshold_adjustments:
            optimized_threshold = threshold_adjustments[reverse_pair]
            self.logger.info(f"Using optimized threshold {optimized_threshold} for {reverse_pair}")
            return optimized_threshold
        
        # Default threshold
        return default_threshold
    
    def validate_embedding_quality(
        self,
        original_embeddings: np.ndarray,
        transformed_embeddings: np.ndarray,
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """
        Validate the quality of transformed embeddings.
        
        Args:
            original_embeddings: Original embeddings
            transformed_embeddings: Transformed embeddings
            sample_size: Number of samples to use for validation
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Sample embeddings for validation
            n_samples = min(sample_size, len(original_embeddings))
            indices = np.random.choice(len(original_embeddings), n_samples, replace=False)
            
            orig_sample = original_embeddings[indices]
            trans_sample = transformed_embeddings[indices]
            
            # Calculate similarity preservation
            orig_similarities = self._calculate_pairwise_similarities(orig_sample)
            trans_similarities = self._calculate_pairwise_similarities(trans_sample)
            
            # Correlation between similarity matrices
            correlation = np.corrcoef(orig_similarities.flatten(), trans_similarities.flatten())[0, 1]
            
            # Mean squared error of similarities
            mse = np.mean((orig_similarities - trans_similarities) ** 2)
            
            return {
                "similarity_correlation": correlation,
                "similarity_mse": mse,
                "dimension_reduction_ratio": trans_sample.shape[1] / orig_sample.shape[1],
                "quality_score": correlation * (1 - mse)  # Combined quality metric
            }
            
        except Exception as e:
            self.logger.error(f"Embedding quality validation failed: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_pairwise_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarities"""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        
        # Calculate cosine similarities
        similarities = np.dot(normalized, normalized.T)
        
        return similarities
