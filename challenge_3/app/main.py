import logging
import logging.config
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from src.model import RegressorModel

load_dotenv()

ROOT_LEVEL = os.environ.get("PROD", "INFO")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "()": "uvicorn.logging.DefaultFormatter",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "level": ROOT_LEVEL,
            "handlers": ["default"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "DEBUG",
            "handlers": ["default"],
        },
        "uvicorn.access": {
            "level": "DEBUG",
            "handlers": ["default"],
        },
    },
}
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)
logger.debug(LOGGING_CONFIG)


class Diamond(BaseModel):
    carat: float = Field(..., gt=0, description="Carat weight of the diamond")
    cut: str = Field(
        ...,
        description="Cut quality of the diamond (Fair, Good, Very Good, Premium, Ideal)",
    )
    color: str = Field(..., description="Color of the diamond (D to Z)")
    clarity: str = Field(..., description="Clarity of the diamond (FL to I3)")
    depth: float = Field(..., gt=0, description="Depth of the diamond")
    table: float = Field(..., gt=0, description="Table of the diamond")
    x: float = Field(..., gt=0, description="X dimension of the diamond")
    y: float = Field(..., gt=0, description="Y dimension of the diamond")
    z: float = Field(..., gt=0, description="Z dimension of the diamond")
    cut_category: Optional[int] = None
    color_category: Optional[int] = None
    clarity_category: Optional[int] = None
    z_depth: Optional[float] = None
    table_width: Optional[float] = None

    def __init__(self, **data):
        super().__init__(**data)

        self.cut_category = self.cut_mapping(self.cut)
        self.color_category = self.color_mapping(self.color)
        self.clarity_category = self.clarity_mapping(self.clarity)

        try:
            self.z_depth = self.depth / (self.z * 100)
        except:
            self.z_depth = 0

        try:
            self.table_width = self.table / (self.x * 100)
        except:
            self.table_width = 0

    @staticmethod
    def cut_mapping(cut: str):
        cut_mapping = {
            "fair": 1,
            "good": 2,
            "very good": 3,
            "premium": 4,
            "ideal": 5,
        }
        return cut_mapping.get(cut.lower(), 0)

    @staticmethod
    def color_mapping(color: str):
        color_grading_scale = list(map(chr, range(ord("D"), ord("Z") + 1)))[
            ::-1
        ]
        color_mapping = {k: v for v, k in enumerate(color_grading_scale, 1)}
        return color_mapping.get(color.upper(), 0)

    @staticmethod
    def clarity_mapping(clarity: str):
        clarity_grading_scale = [
            "FL",
            "IF",
            "VVS1",
            "VVS2",
            "VS1",
            "VS2",
            "SI1",
            "SI2",
            "I1",
            "I2",
            "I3",
        ][::-1]
        clarity_mapping = {
            k: v for v, k in enumerate(clarity_grading_scale, 1)
        }
        return clarity_mapping.get(clarity.upper(), 0)


model = RegressorModel(os.getenv("MODEL_PATH"))


app = FastAPI()


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.post(
    "/price",
    summary="Return the price prediction of a diamond based on its features",
)
def return_price_prediction(
    request: Diamond,
):
    """Returns the price prediction of a diamond based on its features.

    Args:
        request (Diamond): The diamond object containing the features of the diamond.

    Returns:
        dict: A dictionary containing the predicted price of the diamond.
    """
    try:
        input_array = model.preprocess_input(request.model_dump())

        logger.debug(f"Input array: {input_array}")

        price_prediction = model.predict(input_array)

        logger.debug(f"Price prediction: {price_prediction}")

        return {"price": float(price_prediction[0])}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Error predicting the price of the diamond. Please check the input data. Error: {str(e)}"
            },
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, port=port)
