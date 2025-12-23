from fastapi import FastAPI

app = FastAPI()


@app.get("/test")
async def test():
    return {"message": "Test works!"}


@app.get("/stats")
async def stats():
    return {"message": "Stats works!"}
