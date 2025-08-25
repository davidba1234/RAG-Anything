"""VLM (Vision-Language Model) service for image analysis.

This module provides vision-language model capabilities using OpenAI's GPT-4o
or other vision-enabled models for analyzing images, extracting structured data,
and performing visual question answering.
"""

import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from PIL import Image

from app.config import settings


class VLMService:
    """Service for Vision-Language Model operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize VLM service.
        
        Args:
            api_key: OpenAI API key for vision models
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.default_model = "gpt-4o"  # GPT-4o has vision capabilities
        
        if not self.api_key:
            logger.warning("No OpenAI API key found for VLM service")
    
    async def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail",
        model: Optional[str] = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Analyze an image using Vision-Language Model.
        
        Args:
            image_path: Path to the image file
            prompt: Question or instruction about the image
            model: Model to use (default: gpt-4o)
            max_tokens: Maximum tokens in response
            
        Returns:
            Analysis result with description and metadata
        """
        try:
            if not self.api_key:
                return {
                    "success": False,
                    "error": "No API key configured for vision model",
                    "fallback": "OCR-only mode"
                }
            
            # Read and encode image
            image_data = self._encode_image(image_path)
            if not image_data:
                return {
                    "success": False,
                    "error": f"Failed to read image: {image_path}"
                }
            
            # Prepare the vision API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": model or self.default_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result["choices"][0]["message"]["content"]
                    
                    return {
                        "success": True,
                        "analysis": analysis,
                        "model": model or self.default_model,
                        "image_path": image_path,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    error_msg = f"Vision API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg
                    }
                    
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_multiple_images(
        self,
        image_paths: List[str],
        prompt: str = "Compare and describe these images",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze multiple images together.
        
        Args:
            image_paths: List of image file paths
            prompt: Question about the images
            model: Model to use
            
        Returns:
            Comparative analysis result
        """
        try:
            if not self.api_key:
                return {
                    "success": False,
                    "error": "No API key configured for vision model"
                }
            
            # Prepare content with multiple images
            content = [{"type": "text", "text": prompt}]
            
            for path in image_paths:
                image_data = self._encode_image(path)
                if image_data:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": model or self.default_model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 800
            }
            
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result["choices"][0]["message"]["content"]
                    
                    return {
                        "success": True,
                        "analysis": analysis,
                        "images_analyzed": len(image_paths),
                        "model": model or self.default_model
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Vision API error: {response.status_code}"
                    }
                    
        except Exception as e:
            logger.error(f"Multi-image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def extract_structured_data(
        self,
        image_path: str,
        data_type: str = "auto"
    ) -> Dict[str, Any]:
        """Extract structured data from images (tables, charts, etc).
        
        Args:
            image_path: Path to image
            data_type: Type of data to extract (table, chart, diagram, auto)
            
        Returns:
            Structured data extraction result
        """
        prompts = {
            "table": "Extract all table data from this image as JSON. Include column headers and all rows.",
            "chart": "Describe this chart/graph. Extract the data points, axes labels, and trends.",
            "diagram": "Describe this diagram's components and their relationships.",
            "auto": "Identify and extract any structured data (tables, charts, diagrams) from this image."
        }
        
        prompt = prompts.get(data_type, prompts["auto"])
        result = await self.analyze_image(image_path, prompt, max_tokens=1000)
        
        if result.get("success"):
            result["data_type"] = data_type
            
            # Try to parse structured data if it looks like JSON
            analysis = result.get("analysis", "")
            if "{" in analysis and "}" in analysis:
                try:
                    # Extract JSON from the response
                    import re
                    json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
                    if json_match:
                        structured_data = json.loads(json_match.group())
                        result["structured_data"] = structured_data
                except:
                    pass
        
        return result
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string or None if failed
        """
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Open and potentially resize image if too large
            with Image.open(path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize if too large (max 2048 pixels on longest side)
                max_size = 2048
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                
                # Encode to base64
                return base64.b64encode(buffer.read()).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    async def find_relevant_images(
        self,
        kb_path: str,
        query: str,
        max_images: int = 5
    ) -> List[str]:
        """Find relevant images in knowledge base for a query.
        
        Args:
            kb_path: Knowledge base directory path
            query: User query
            max_images: Maximum number of images to return
            
        Returns:
            List of relevant image paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        images = []
        
        try:
            kb_dir = Path(kb_path)
            if kb_dir.exists():
                for ext in image_extensions:
                    images.extend(kb_dir.rglob(f"*{ext}"))
                    images.extend(kb_dir.rglob(f"*{ext.upper()}"))
            
            # For now, return most recent images
            # In production, you'd use semantic search or metadata
            images.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            return [str(img) for img in images[:max_images]]
            
        except Exception as e:
            logger.error(f"Failed to find images: {e}")
            return []


# Global VLM service instance
vlm_service = None


def get_vlm_service() -> VLMService:
    """Get or create VLM service instance."""
    global vlm_service
    if vlm_service is None:
        vlm_service = VLMService()
    return vlm_service