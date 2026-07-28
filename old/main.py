"""Super Secret project"""
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class Position:
    """Represents a position in 3D space."""
    x: float = 0
    y: float = 0
    z: float = 0


class PositionPayload(BaseModel):
    """JSON body schema for setting a new position."""
    x: float
    y: float
    z: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.position = Position()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    """Root endpoint returning the status of the API."""
    return {"Status": "OK"}


@app.get("/position")
def get_position():
    """Endpoint to get the current position."""
    p = app.state.position
    return {"position": {"x": p.x, "y": p.y, "z": p.z}}


@app.post("/newPos")
def new_position(payload: PositionPayload):
    """Endpoint to set a new position."""
    app.state.position = Position(payload.x, payload.y, payload.z)
    p = app.state.position
    return {"position": {"x": p.x, "y": p.y, "z": p.z}}


@app.post("/feed")
def feed():
    """Endpoint to receive feed data."""
    return {"message": "Feed received"}