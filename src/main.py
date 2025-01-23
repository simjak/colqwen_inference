import base64
import io
from typing import Literal
from uuid import UUID

from modal import Image as ModalImage
from modal import Mount, Secret, asgi_app
from PIL import Image as PILImage
from pydantic import BaseModel

from colqwen_inference.colqwen import colqwen_model
from colqwen_inference.modal_app import app
from src.utils.logger import logger

img = (
    ModalImage.debian_slim(python_version="3.12.0")
    .pip_install(
        "openai>=1.59.3",
        "opencv-python-headless>=4.10.0.84",
        "pydantic>=2.10.4",
        "fastapi[standard]>=0.115.6",
        "pillow",
        "qdrant-client>=1.13.0",
    )
    .pip_install("numpy>=1.24.0")
)


class EmbeddingRequest(BaseModel):
    model: Literal["colqwen"]
    # Text string or list of Base64-encoded strings (instead of raw bytes)
    input: str | list[str]
    input_type: Literal["text", "image"]
    batch_size: int = 1


class EmbeddingResponse(BaseModel):
    model: str
    embeddings: (
        list[list[list[float]]] | None
    )  # [batch_size, sequence_length, hidden_size]


class SearchRequest(BaseModel):
    query: str
    instance_id: UUID
    count: int = 3


@app.function(
    image=img,
    mounts=[Mount.from_local_python_packages("colqwen_inference")],
    secrets=[Secret.from_dotenv()],
    concurrency_limit=1,
    container_idle_timeout=300,
    timeout=600,
    allow_concurrent_inputs=10,
)
@asgi_app()
def web():
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(title="Embedding Service")

    origins = ["*"]

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/embed", response_model=EmbeddingResponse)
    async def create_embedding(request: EmbeddingRequest):
        try:
            if request.input_type == "text":
                if not isinstance(request.input, str):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Text input must be a string",
                    )
                # Use .remote.aio(...) for a regular async modal method
                # Get text embeddings and convert to numpy array
                embeddings = await colqwen_model.embed_queries.remote.aio(
                    [request.input]
                )
                # embeddings = np.array(embeddings)  # Convert to numpy array

            else:
                # Expect a list of Base64-encoded image strings
                if not isinstance(request.input, list):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Image input must be a list of Base64-encoded strings",
                    )

                # Decode Base64-encoded strings to PIL Images
                pil_images = []
                logger.info(f"Decoding {len(request.input)} images")
                for encoded_img_str in request.input:
                    try:
                        # Decode base64 to bytes
                        img_bytes = base64.b64decode(encoded_img_str)
                        # Convert bytes to PIL Image
                        img = PILImage.open(io.BytesIO(img_bytes))
                        # Convert to RGB if necessary (handles PNG transparency)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        pil_images.append(img)
                    except Exception as e:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid image data: {str(e)}",
                        )
                logger.info(f"Decoded {len(pil_images)} images")

                # Collect image embeddings
                embeddings = await colqwen_model.embed_images.remote.aio(
                    pil_images, request.batch_size
                )

            return EmbeddingResponse(model=request.model, embeddings=embeddings)

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    return web_app
