from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel


ALLOWED_OPERAS = {
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Pluna",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines",
}

app = FastAPI()
model = DelayModel()


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def validate_opera(cls, value: str) -> str:
        if value not in ALLOWED_OPERAS:
            raise ValueError("Invalid OPERA value")
        return value

    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, value: str) -> str:
        if value not in {"I", "N"}:
            raise ValueError("Invalid TIPOVUELO value")
        return value

    @validator("MES")
    def validate_mes(cls, value: int) -> int:
        if value < 1 or value > 12:
            raise ValueError("Invalid MES value")
        return value


class PredictRequest(BaseModel):
    flights: List[Flight]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


def _load_model_if_needed() -> None:
    if model._model is not None:
        return

    data_path = Path(__file__).resolve().parents[1] / "data" / "data.csv"
    data = pd.read_csv(data_path)
    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features=features, target=target)


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    _load_model_if_needed()

    flights_df = pd.DataFrame([flight.dict() for flight in request.flights])
    features = model.preprocess(data=flights_df)
    predictions = model.predict(features=features)

    return {"predict": predictions}