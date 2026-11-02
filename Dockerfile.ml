FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt requirements-analytics.txt requirements-dl.txt requirements-ml.txt /app/
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY . /app

EXPOSE 8000

# One worker: each process holds its own copy of the models in memory. Requests run in a
# thread pool, so one slow LLM call does not block the others.
CMD ["uvicorn", "ml_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
