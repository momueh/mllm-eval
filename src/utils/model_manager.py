import base64
import io
import logging
from typing import Dict, Any, Optional
from PIL import Image

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from concurrent.futures import ThreadPoolExecutor, as_completed

from .api_manager import APIManager

logger = logging.getLogger("model_manager")


class ModelManager:
    def __init__(
        self,
        config: Dict[str, Any],
        api_manager: APIManager,
    ):
        self.config = config
        self.api_manager = api_manager

    def _encode_image(self, image_path: str) -> Dict[str, Any]:
        """Encode image for all model formats."""
        # Load image
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()

        # Get base64 for all formats
        base64_data = base64.b64encode(binary_data).decode("utf-8")

        # For Anthropic, get additional info from PIL
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            anthropic_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            media_type = "image/png"

        return {
            "base64_standard": base64_data,
            "anthropic": {"base64": anthropic_data, "media_type": media_type},
        }

    def _encode_image_object(self, image) -> Dict[str, Any]:
        """Encode PIL image object for all model formats."""
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        binary_data = buffer.getvalue()

        # Get base64 for all formats
        base64_data = base64.b64encode(binary_data).decode("utf-8")

        # For Anthropic format
        anthropic_data = base64_data
        media_type = "image/png"

        return {
            "base64_standard": base64_data,
            "anthropic": {"base64": anthropic_data, "media_type": media_type},
        }

    def query_openai(self, prompt: str, image) -> str:
        """Query OpenAI model with image and prompt."""
        # Apply rate limiting
        self.api_manager.wait_if_needed("openai")

        # Encode image
        # Handle different image input types
        if isinstance(image, str):
            # It's a file path
            image_data = self._encode_image(image)
        else:
            # It's a PIL Image or similar object
            image_data = self._encode_image_object(image)

        base64_image = image_data["base64_standard"]

        # Initialize model
        client = ChatOpenAI(
            model=self.config["openai"]["model_name"],
            api_key=self.config["openai"]["api_key"],
        )

        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        # Get response
        logger.info(f"Querying OpenAI model: {self.config['openai']['model_name']}")
        response = client.invoke([message])
        return response.content

    def query_anthropic(self, prompt: str, image) -> str:
        """Query Anthropic model with image and prompt."""
        # Apply rate limiting
        self.api_manager.wait_if_needed("anthropic")

        # Encode image
        # Handle different image input types
        if isinstance(image, str):
            # It's a file path
            image_data = self._encode_image(image)
        else:
            # It's a PIL Image or similar object
            image_data = self._encode_image_object(image)

        base64_image = image_data["anthropic"]["base64"]
        media_type = image_data["anthropic"]["media_type"]

        # Initialize model
        client = ChatAnthropic(
            anthropic_api_key=self.config["anthropic"]["api_key"],
            model_name=self.config["anthropic"]["model_name"],
        )

        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                },
            ]
        )

        # Get response
        logger.info(
            f"Querying Anthropic model: {self.config['anthropic']['model_name']}"
        )
        response = client.invoke([message])
        return response.content

    def query_gemini(self, prompt: str, image) -> str:
        """Query Gemini model with image and prompt."""
        # Apply rate limiting
        self.api_manager.wait_if_needed("gemini")

        # Encode image
        # Handle different image input types
        if isinstance(image, str):
            # It's a file path
            image_data = self._encode_image(image)
        else:
            # It's a PIL Image or similar object
            image_data = self._encode_image_object(image)

        base64_image = image_data["base64_standard"]

        # Initialize model
        client = ChatGoogleGenerativeAI(
            google_api_key=self.config["gemini"]["api_key"],
            model=self.config["gemini"]["model_name"],
        )

        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        # Get response
        logger.info(f"Querying Gemini model: {self.config['gemini']['model_name']}")
        response = client.invoke([message])
        return response.content

    def query_all_models(self, prompt, image_path, parallel=True, max_workers=None):
        """
        Query all models with the same prompt and image.

        Args:
            prompt: The text prompt to send
            image: Either a file path (str) or a PIL Image object
        """

        results = {}
        model_names = ["openai", "anthropic", "gemini"]

        # Use ThreadPoolExecutor for parallel execution

        with ThreadPoolExecutor(max_workers=len(model_names)) as executor:
            # Create a mapping of futures to model names
            future_to_model = {}

            # Submit tasks for each model
            for model_name in model_names:
                if model_name == "openai":
                    future = executor.submit(self.query_openai, prompt, image_path)
                elif model_name == "anthropic":
                    future = executor.submit(self.query_anthropic, prompt, image_path)
                elif model_name == "gemini":
                    future = executor.submit(self.query_gemini, prompt, image_path)

                future_to_model[future] = model_name

            # Process results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    results[model_name] = future.result()
                except Exception as e:
                    logger.error(f"Error querying {model_name}: {str(e)}")
                    results[model_name] = f"Error: {str(e)}"

        return results

    # try:
    #     results["openai"] = self.query_openai(prompt, image)
    # except Exception as e:
    #     logger.error(f"Error querying OpenAI: {str(e)}")
    #     results["openai"] = f"Error: {str(e)}"

    # try:
    #     results["anthropic"] = self.query_anthropic(prompt, image)
    # except Exception as e:
    #     logger.error(f"Error querying Anthropic: {str(e)}")
    #     results["anthropic"] = f"Error: {str(e)}"

    # try:
    #     results["gemini"] = self.query_gemini(prompt, image)
    # except Exception as e:
    #     logger.error(f"Error querying Gemini: {str(e)}")
    #     results["gemini"] = f"Error: {str(e)}"

    # return results
