"""
Transfer Utility Functions

This module provides utility functions for knowledge graph transfer operations.
"""

import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

from ..utils import logger


def generate_transfer_id() -> str:
    """Generate a unique transfer ID using existing LightRAG hash utilities"""
    from ..utils import compute_mdhash_id
    import time

    # Use LightRAG's existing hash function for consistency
    timestamp = str(int(time.time()))
    return compute_mdhash_id(f"transfer_{timestamp}", prefix="xfer-")


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate file hash using existing LightRAG utilities"""
    from ..utils import compute_mdhash_id

    # Read file content and use existing hash function for consistency
    with open(file_path, "rb") as f:
        content = f.read()

    return compute_mdhash_id(content.decode('utf-8', errors='ignore'), prefix="file-")


def get_directory_size(directory: Union[str, Path]) -> int:
    """Get total size of directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def validate_transfer_package(package_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate transfer package structure and integrity.
    
    Args:
        package_path: Path to transfer package
        
    Returns:
        Dictionary with validation results
    """
    package_path = Path(package_path)
    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "structure": {},
        "size": 0
    }
    
    try:
        if not package_path.exists():
            result["errors"].append("Transfer package does not exist")
            return result
        
        # Check if it's a compressed package
        if package_path.suffix in ['.gz', '.zip'] or package_path.name.endswith('.tar.gz'):
            result["structure"]["compressed"] = True
            result["size"] = package_path.stat().st_size

            # Basic validation for compressed packages
            try:
                import tarfile
                import zipfile

                if package_path.name.endswith('.tar.gz') or package_path.suffix == '.gz':
                    # Validate tar.gz file
                    with tarfile.open(package_path, 'r:gz') as tar:
                        members = tar.getnames()
                        result["structure"]["archive_members"] = len(members)

                        # Check for required files in archive
                        has_metadata = any('metadata.json' in member for member in members)
                        if has_metadata:
                            result["structure"]["metadata.json"] = True
                        else:
                            result["errors"].append("Missing metadata.json in compressed package")

                elif package_path.suffix == '.zip':
                    # Validate zip file
                    with zipfile.ZipFile(package_path, 'r') as zip_file:
                        members = zip_file.namelist()
                        result["structure"]["archive_members"] = len(members)

                        # Check for required files in archive
                        has_metadata = any('metadata.json' in member for member in members)
                        if has_metadata:
                            result["structure"]["metadata.json"] = True
                        else:
                            result["errors"].append("Missing metadata.json in compressed package")

            except Exception as e:
                result["errors"].append(f"Failed to validate compressed package: {str(e)}")
        else:
            # Check directory structure
            if not package_path.is_dir():
                result["errors"].append("Transfer package is not a directory")
                return result
            
            result["size"] = get_directory_size(package_path)
            
            # Check for required files
            required_files = ["metadata.json"]
            for req_file in required_files:
                file_path = package_path / req_file
                if not file_path.exists():
                    result["errors"].append(f"Missing required file: {req_file}")
                else:
                    result["structure"][req_file] = True
            
            # Check for optional directories
            optional_dirs = ["structure", "embeddings", "cache", "documents"]
            for opt_dir in optional_dirs:
                dir_path = package_path / opt_dir
                if dir_path.exists() and dir_path.is_dir():
                    result["structure"][opt_dir] = True
                    result["structure"][f"{opt_dir}_size"] = get_directory_size(dir_path)
        
        # If no errors, mark as valid
        if not result["errors"]:
            result["valid"] = True
        
    except Exception as e:
        result["errors"].append(f"Validation error: {str(e)}")
    
    return result


def create_transfer_manifest(
    package_path: Union[str, Path],
    source_config: Dict[str, Any],
    target_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive manifest for the transfer package.
    
    Args:
        package_path: Path to transfer package
        source_config: Source LightRAG configuration
        target_config: Target LightRAG configuration (optional)
        
    Returns:
        Manifest dictionary
    """
    package_path = Path(package_path)
    
    manifest = {
        "transfer_id": generate_transfer_id(),
        "created_at": str(asyncio.get_event_loop().time()),
        "package_path": str(package_path),
        "package_size": get_directory_size(package_path) if package_path.is_dir() else package_path.stat().st_size,
        "source_config": source_config,
        "target_config": target_config,
        "files": [],
        "checksums": {}
    }
    
    # Add file information
    if package_path.is_dir():
        for file_path in package_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(package_path)
                file_info = {
                    "path": str(relative_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                }
                manifest["files"].append(file_info)
                
                # Calculate checksum for important files
                if file_path.suffix in ['.json', '.csv', '.xlsx', '.graphml']:
                    try:
                        manifest["checksums"][str(relative_path)] = calculate_file_hash(file_path)
                    except Exception as e:
                        logger.warning(f"Could not calculate checksum for {file_path}: {e}")
    
    return manifest


def compare_configurations(
    source_config: Dict[str, Any],
    target_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare source and target configurations to identify differences.
    
    Args:
        source_config: Source LightRAG configuration
        target_config: Target LightRAG configuration
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "identical": True,
        "differences": [],
        "compatibility_issues": [],
        "recommendations": []
    }
    
    # Compare storage backends
    storage_types = ["graph_storage", "vector_storage", "kv_storage", "doc_status_storage"]
    for storage_type in storage_types:
        source_val = source_config.get(storage_type)
        target_val = target_config.get(storage_type)
        
        if source_val != target_val:
            comparison["identical"] = False
            comparison["differences"].append({
                "field": storage_type,
                "source": source_val,
                "target": target_val
            })
            
            # Check for compatibility issues
            if storage_type == "graph_storage":
                if source_val == "NetworkXStorage" and target_val != "NetworkXStorage":
                    comparison["compatibility_issues"].append(
                        "GraphML format may not be directly compatible with target graph storage"
                    )
    
    # Compare embedding configurations
    source_embedding = source_config.get("embedding_func", {})
    target_embedding = target_config.get("embedding_func", {})
    
    if source_embedding.get("embedding_dim") != target_embedding.get("embedding_dim"):
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "embedding_dimension",
            "source": source_embedding.get("embedding_dim"),
            "target": target_embedding.get("embedding_dim")
        })
        comparison["compatibility_issues"].append(
            "Different embedding dimensions require regeneration of all embeddings"
        )
        comparison["recommendations"].append(
            "Use regenerate_embeddings=True in transfer configuration"
        )
    
    # Compare performance settings
    performance_fields = [
        "chunk_token_size",
        "chunk_overlap_token_size", 
        "entity_extract_max_gleaning",
        "max_async",
        "max_tokens_for_global_context"
    ]
    
    for field in performance_fields:
        source_val = source_config.get(field)
        target_val = target_config.get(field)
        
        if source_val != target_val and source_val is not None and target_val is not None:
            comparison["differences"].append({
                "field": field,
                "source": source_val,
                "target": target_val
            })
    
    return comparison


async def estimate_transfer_time(
    package_size: int,
    network_speed_mbps: float = 100.0,
    processing_overhead: float = 1.5
) -> Dict[str, float]:
    """
    Estimate transfer and processing time.
    
    Args:
        package_size: Size of transfer package in bytes
        network_speed_mbps: Network speed in Mbps
        processing_overhead: Processing overhead multiplier
        
    Returns:
        Dictionary with time estimates
    """
    # Convert to MB
    package_size_mb = package_size / (1024 * 1024)
    
    # Estimate network transfer time
    network_time_seconds = (package_size_mb * 8) / network_speed_mbps
    
    # Estimate processing time (extraction, import, validation)
    processing_time_seconds = network_time_seconds * processing_overhead
    
    # Total time
    total_time_seconds = network_time_seconds + processing_time_seconds
    
    return {
        "network_transfer_seconds": network_time_seconds,
        "processing_seconds": processing_time_seconds,
        "total_seconds": total_time_seconds,
        "total_minutes": total_time_seconds / 60,
        "total_hours": total_time_seconds / 3600
    }


def create_transfer_script(
    source_config: Dict[str, Any],
    target_config: Dict[str, Any],
    package_path: str,
    output_path: str = "transfer_script.py"
) -> str:
    """
    Generate a Python script for executing the transfer.
    
    Args:
        source_config: Source configuration
        target_config: Target configuration
        package_path: Path to transfer package
        output_path: Output path for script
        
    Returns:
        Path to generated script
    """
    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated LightRAG Knowledge Graph Transfer Script

This script was generated to transfer a knowledge graph from:
Source: {source_config.get('working_dir', 'unknown')}
Target: {target_config.get('working_dir', 'unknown')}

Package: {package_path}
"""

import asyncio
from lightrag import LightRAG, KnowledgeGraphTransfer
from lightrag.transfer import TransferConfig

# Target LightRAG configuration
target_rag = LightRAG(
    working_dir="{target_config.get('working_dir', './rag_storage_target')}",
    graph_storage="{target_config.get('graph_storage', 'NetworkXStorage')}",
    vector_storage="{target_config.get('vector_storage', 'NanoVectorDBStorage')}",
    kv_storage="{target_config.get('kv_storage', 'JsonKVStorage')}",
    # Add your LLM and embedding functions here
    # llm_model_func=your_llm_function,
    # embedding_func=your_embedding_function,
)

async def main():
    # Create transfer instance
    transfer = KnowledgeGraphTransfer()
    
    # Configure transfer
    config = TransferConfig(
        include_embeddings=False,  # Regenerate with target model
        include_llm_cache=True,
        include_documents=True,
        verify_integrity=True,
        regenerate_embeddings=True
    )
    
    # Execute transfer
    print("Starting knowledge graph transfer...")
    success = await transfer.import_from_transfer(
        target_rag=target_rag,
        transfer_package_path="{package_path}",
        config=config
    )
    
    if success:
        print("Transfer completed successfully!")
        
        # Test query functionality
        test_query = "What are the main topics in this knowledge base?"
        result = target_rag.query(test_query)
        print(f"Test query result: {{result[:200]}}...")
        
    else:
        print("Transfer failed. Check logs for details.")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_path, 0o755)
    
    return output_path


def cleanup_transfer_files(
    package_path: Union[str, Path],
    keep_original: bool = True
) -> bool:
    """
    Clean up temporary transfer files.
    
    Args:
        package_path: Path to transfer package
        keep_original: Whether to keep the original package
        
    Returns:
        True if cleanup successful
    """
    try:
        package_path = Path(package_path)
        
        # Clean up extracted directories (temporary)
        temp_patterns = ["*_extracted", "*_temp", "lightrag_transfer_*"]
        
        for pattern in temp_patterns:
            for temp_path in package_path.parent.glob(pattern):
                if temp_path.is_dir():
                    import shutil
                    shutil.rmtree(temp_path, ignore_errors=True)
                    logger.info(f"Cleaned up temporary directory: {temp_path}")
        
        # Optionally remove original package
        if not keep_original and package_path.exists():
            if package_path.is_dir():
                import shutil
                shutil.rmtree(package_path)
            else:
                package_path.unlink()
            logger.info(f"Removed original package: {package_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return False
