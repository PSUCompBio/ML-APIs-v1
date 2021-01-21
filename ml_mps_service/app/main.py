from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.predict_mps import mps

app = FastAPI(openapi_url="/api/predict/openapi.json", docs_url="/api/predict/docs")

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(mps, prefix='/api/predict', tags=['maximum principle strain'])




