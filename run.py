import uvicorn
import os

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",
        port=5000,
        reload=False
    )
