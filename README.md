# Serverless ColQwen Inference Service

A serverless FastAPI service for generating embeddings using ColQwen-compatible models, deployed on [Modal](https://modal.com/) platform.
This project builds upon the work from [Vision Is All You Need](https://github.com/Softlandia-Ltd/vision-is-all-you-need), adapting and extending its capabilities for embedding generation.

## Features

- Single `/embed` endpoint for both text and image embeddings
- Support for ColQwen-compatible models
- Batch processing for efficient image embedding generation
- GPU acceleration with A10G on Modal platform
- Serverless deployment with automatic scaling
- Simple Python client included

## Installation

1. Install [uv](https://github.com/astral-sh/uv) for dependency management:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/softlandia/colqwen-inference.git
cd colqwen-inference
```

3. Install dependencies:
```bash
uv sync
```

## Environment Setup

1. Create a `.env` file in the project root:
```bash
HF_TOKEN=your_huggingface_token  # Required for model access
```

2. For development, install additional dependencies:
```bash
uv sync --group dev
```

## Usage

### Running Locally

Start the service locally using Modal:
```bash
modal serve src/main.py
```

### Deployment

Deploy to Modal:
```bash
modal deploy src/main.py
```

### Python Client

The package includes a simple client for easy integration:

```python
from colqwen_inference.client import ColpaliClient

# Initialize client
client = ColpaliClient(base_url="your_modal_endpoint")

# Generate text embeddings
text_embedding = await client.embed_text("Your text here")

# Generate image embeddings
import base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

image_embeddings = await client.embed_images(
    [image_base64],
    batch_size=1
)
```

### REST API

The service exposes a single endpoint for both text and image embeddings:

**POST `/embed`**

Request body:
```json
{
    "model": "colqwen",
    "input": "text string | list of base64 encoded images",
    "input_type": "text | image",
    "batch_size": 1  // Optional, defaults to 1
}
```

Response:
```json
{
    "model": "string",
    "embeddings": [[float]]  // List of embedding vectors
}
```

## Project Structure

```
colqwen-inference/
├── examples/                 # Example notebooks and data
│   └── 00_colqwen_embeddings.ipynb
├── src/
│   ├── colqwen_inference/   # Core package
│   │   ├── colqwen.py      # Model implementation
│   │   └── modal_app.py    # Modal service definition
│   ├── utils/              # Utility functions
│   ├── client.py           # Python client
│   └── main.py             # Service entry point
├── pyproject.toml          # Project configuration
└── README.md
```

## Credits

This project builds upon and extends the work from [Vision Is All You Need](https://github.com/Softlandia-Ltd/vision-is-all-you-need).
I'm grateful to the original authors for their contributions to the field.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
