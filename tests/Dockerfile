# Build and run this Dockerfile which won't work inside a container: (NOT TESTED YET)
# docker build -f tests/Dockerfile -t network-tests .
# docker run --rm -v $(pwd)/reports:/app/reports network-tests

FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY ../ ./

RUN poetry install

CMD ["poetry", "run", "pytest", "--junitxml=report.xml"]