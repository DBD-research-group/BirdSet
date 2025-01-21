# Build and run this Dockerfile: (NOT TESTED YET)
# docker build -f tests/Dockerfile -t network-tests .
# docker run --rm -v $(pwd)/reports:/app/reports network-tests

FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY ../ ./

RUN poetry install --no-root

CMD ["poetry", "run", "pytest", "--junitxml=report.xml"]
