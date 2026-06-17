"""Super Secret project"""
from fastapi import FastAPI
from enum import Enum

class Position(Enum):
    """Represents a position in 3D space."""
    X = 0
    Y = 0
    Z = 0

position = [Position.X, Position.Y, Position.Z]

app = FastAPI()

@app.get("/")
def root():
    """Root endpoint returning the status of the API."""
    return {"Status": "OK"}

@app.get("/position")
def get_position():
    """Endpoint to get the current position."""
    return {"position": [p.value for p in position]}

@app.get("/newPos")
def new_position(x: int, y: int, z: int):
    """Endpoint to set a new position."""
    global position
    position = [Position.X, Position.Y, Position.Z]
    position[0] = Position(x)
    position[1] = Position(y)
    position[2] = Position(z)
    return {"position": [p.value for p in position]}

@app.post("/feed")
def feed():
    """Endpoint to receive feed data."""
    return {"message": "Feed received"}