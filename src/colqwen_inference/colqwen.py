"""
ColQwen Model Implementation

This module provides a Modal-based implementation of the ColQwen2 model for generating
embeddings from text and images. The model is deployed on Modal's infrastructure and
uses GPU acceleration when available.

The implementation includes:
- Model initialization and loading
- Text query processing and embedding generation
- Image processing and embedding generation
- Batch processing capabilities for both text and images
"""

import os
from typing import cast

import modal

from colqwen_inference.modal_app import app
from src.utils.logger import logger

CACHE_DIR = "/hf-cache"

# Define the base Modal image with required dependencies
img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "colpali_engine",  # Core model engine
        "torch",           # PyTorch for deep learning
        "transformers",    # Hugging Face Transformers
        "einops",         # Array operations
        "vidore_benchmark",
        "pillow",         # Image processing
    )
    .pip_install("numpy")  # Numerical computing
    .pip_install("opencv_python_headless")  # OpenCV for image processing
)


@app.cls(
    gpu="A10G",  # Use A10G GPU for acceleration
    secrets=[modal.Secret.from_dotenv()],
    cpu=4,
    timeout=600,
    container_idle_timeout=300,
    image=img,
)
class ColQwenModel:
    """
    Modal-based ColQwen2 model implementation for generating embeddings.

    This class handles the initialization, loading, and inference of the ColQwen2 model,
    providing methods for both text and image embedding generation.
    """

    def __init__(self):
        """Initialize the ColQwen model with required components."""
        from colpali_engine.models import ColQwen2Processor
        from transformers import PreTrainedModel

        self.model_name = "vidore/colqwen2-v1.0"
        self.model: PreTrainedModel
        self.token = os.environ.get("HF_TOKEN")
        self.processor: ColQwen2Processor

    @modal.build()
    @modal.enter()
    def load_model(self):
        """
        Load the ColQwen2 model and processor.

        This method is called when the container starts. It:
        1. Checks for GPU availability
        2. Sets appropriate device and precision
        3. Loads the model and processor
        """
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        # Set device and precision based on hardware
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[1] >= 8 * 1024**3:
            device = torch.device("cuda")
            torch_type = torch.bfloat16
        else:
            device = torch.device("cpu")
            torch_type = torch.float32

        # Load model with appropriate settings
        self.model = ColQwen2.from_pretrained(
            self.model_name,
            torch_dtype=torch_type,
            device_map=device,
            token=self.token,
        ).eval()

        # Load processor
        self.processor = cast(
            ColQwen2Processor,
            ColQwen2Processor.from_pretrained(self.model_name),
        )

    def process_images(self, pil_images: list):
        """
        Process a list of PIL images for embedding generation.

        Args:
            pil_images: List of PIL Image objects to process

        Returns:
            Processed image tensors ready for the model
        """
        return self.processor.process_images(pil_images).to(self.model.device)

    def process_queries(
        self,
        queries: list[str],
        max_length: int = 50,
        suffix: str | None = None,
    ):
        """
        Process text queries for embedding generation.

        Args:
            queries: List of text queries to process
            max_length: Maximum sequence length (default: 50)
            suffix: Optional suffix to append to queries

        Returns:
            Processed query tensors ready for the model
        """
        return self.processor.process_queries(
            queries=queries, max_length=max_length, suffix=suffix
        ).to(self.model.device)

    @modal.method()
    async def embed_queries(self, queries: list[str], batch_size: int = 1) -> list:
        """
        Generate embeddings for text queries.

        Args:
            queries: List of text queries to embed
            batch_size: Number of queries to process at once (default: 1)

        Returns:
            List of embedding vectors for each query
        """
        import torch
        from torch.utils.data import DataLoader

        logger.info(f"Started query embeddings, number of queries: {len(queries)}")

        # Create dataloader for batch processing
        dataloader = DataLoader(
            queries,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.process_queries(x),
        )

        all_embeddings: list[list[list[float]]] = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {
                    k: v.to(self.model.device) for k, v in batch_query.items()
                }
                embeddings_query = self.model(**batch_query)
                # Convert tensors to Python lists
                cpu_embeddings = embeddings_query.to("cpu")
                embeddings_list = [
                    tensor.tolist() for tensor in torch.unbind(cpu_embeddings)
                ]
                all_embeddings.extend(embeddings_list)

            logger.info(f"Processed {len(all_embeddings)} query embeddings")

        return all_embeddings

    @modal.method()
    async def embed_images(
        self,
        images: list,
        batch_size: int,
    ) -> list[list[list[float]]] | None:
        """
        Generate embeddings for images.

        Args:
            images: List of PIL Image objects to embed
            batch_size: Number of images to process at once

        Returns:
            List of embedding vectors for each image, or None if processing fails

        Raises:
            Exception: If image processing fails
        """
        logger.info(
            f"Started embeddings, number of images: {len(images)}, batch_size: {batch_size}"
        )
        try:
            import torch
            from torch.utils.data import DataLoader

            # Create dataloader for batch processing
            dataloader = DataLoader(
                images,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: self.process_images(x),
            )

            all_embeddings: list[list[list[float]]] = []
            for batch_doc in dataloader:
                with torch.no_grad():
                    batch_doc = {
                        k: v.to(self.model.device) for k, v in batch_doc.items()
                    }
                    batch_embeddings = self.model(**batch_doc)
                    # Convert tensors to Python lists
                    cpu_embeddings = batch_embeddings.to("cpu")
                    embeddings_list = [
                        tensor.tolist() for tensor in torch.unbind(cpu_embeddings)
                    ]
                    all_embeddings.extend(embeddings_list)

            logger.info(f"Processed {len(all_embeddings)} embeddings")
            return all_embeddings
        except Exception as e:
            logger.error(f"Error embedding images: {e}")
            raise e


# Create singleton instance
colqwen_model = ColQwenModel()
