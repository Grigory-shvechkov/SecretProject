"""Super Secret project"""
from fastapi import FastAPI
from dataclasses import dataclass

@dataclass
class Position:
    """Represents a position in 3D space."""
    x: float = 0
    y: float = 0
    z: float = 0

position = Position()

app = FastAPI()

@app.get("/")
def root():
    """Root endpoint returning the status of the API."""
    return {"Status": "OK"}

@app.get("/position")
def get_position():
    """Endpoint to get the current position."""
    return {"position": {"x": position.x, "y": position.y, "z": position.z}}

@app.post("/newPos")
def new_position(x: int, y: int, z: int):
    """Endpoint to set a new position."""
    global position
    position = Position(x, y, z)
    return {"position": {"x": position.x, "y": position.y, "z": position.z}}

@app.post("/feed")
def feed():
    """Endpoint to receive feed data."""
    return {"message": "Feed received"}