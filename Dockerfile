FROM python:3.12-slim

WORKDIR /app

COPY requirements-ranking.txt .
RUN pip install --no-cache-dir -r requirements-ranking.txt

COPY rank.py validate_submission.py ./
COPY src ./src

ENTRYPOINT ["python", "rank.py"]
