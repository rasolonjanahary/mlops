FROM python:3.12-slim

WORKDIR /app

COPY require.txt .

RUN pip install -r require.txt

COPY . .

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]