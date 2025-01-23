from enum import Enum
from typing import Literal, Union

import aiohttp
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import logger


class InputType(Enum):
    TEXT = "text"
    IMAGE = "image"


class EmbeddingRequest(BaseModel):
    model: Literal["colqwen"]
    input: Union[str, list[str]]
    input_type: InputType
    batch_size: int = 1

    class Config:
        use_enum_values = True


class EmbeddingResponse(BaseModel):
    model: str
    embeddings: list[list[list[float]]]



class ColqwenClient:
    """Async client for interacting with Colqwen Embedding Service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the Colqwen client.

        Args:
            base_url: The base URL of the Colqwen embedding service.
        """
        self.base_url = base_url.rstrip("/")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_text(self, text: str) -> list[list[list[float]]]:
        """Generate embeddings for text using the Colqwen Embeddings API.

        Args:
            text: The input text to generate embeddings for.

        Returns:
            A numpy array containing the embeddings.

        Raises:
            Exception: If there's an error generating embeddings.
        """
        try:
            request_data = EmbeddingRequest(
                model="colqwen", input=text, input_type=InputType.TEXT, batch_size=1
            )

            # Create and use the aiohttp.ClientSession in a context manager
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embed", json=request_data.model_dump()
                ) as response:
                    if response.status != 200:
                        error_detail = await response.text()
                        logger.error(f"Colqwen API error: {error_detail}")
                        response.raise_for_status()

                    data = await response.json()
                    response_model = EmbeddingResponse(**data)

                    return response_model.embeddings

        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_images(
        self, base64_images: list[str], batch_size: int = 1
    ) -> list[list[list[float]]]:
        """Generate embeddings for images using the ColQwen Embeddings API.

        Args:
            base64_images (List[str]): List of base64-encoded images to generate embeddings for.
            batch_size (int, optional): Number of images per batch on the server side. Defaults to 1.

        Returns:
            np.ndarray: A 3D float array with shape [num_images, sequence_length, hidden_size].
        """
        try:
            request_data = EmbeddingRequest(
                model="colqwen",
                input=base64_images,
                input_type=InputType.IMAGE,
                batch_size=batch_size,
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embed", json=request_data.model_dump()
                ) as response:
                    if response.status != 200:
                        error_detail = await response.text()
                        logger.error(f"ColQwen API error: {error_detail}")
                        response.raise_for_status()

                    data = await response.json()
                    response_model = EmbeddingResponse(**data)
                    # Convert server result to a numpy array
                    # embeddings = np.array(response_model.embeddings, dtype=float)
                    return response_model.embeddings

        except Exception as e:
            logger.error(f"Error generating image embeddings: {str(e)}")
            raise

