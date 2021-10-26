# Fastapi + MLflow + streamlit

Setup env. I hope I covered all.
```bash
pip install -r requirements.txt
```
# Start app
Go in the root dir and run these

Streamlit
```bash
streamlit run frontend/streamlit_main.py
```

FastAPI 
```
uvicorn backend.main:app
```

MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///db/bakckend.db
```

## Docker
- Mlflow: http://localhost:5000
- FastApi: http://localhost:8000/docs
- Streamlit: http://localhost:8501/

```bash
docker-compose build
docker-compose up
```

# Architecture
![image](resources/arch.png)

# UI 
![image](resources/train_pic.png)
![image](resources/pred_pic.png)


## TODO
- [x] Dockerize
- [ ] Testing
- [ ] Maybe add celery instead of that background task? (Needs extra configs though)

