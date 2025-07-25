nohup gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8081 --workers 4 --threads 2 > server.log 2>&1 &
