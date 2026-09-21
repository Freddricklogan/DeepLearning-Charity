# Builds the report in a container: docker build -t charity-model . && docker run --rm -v "$PWD/dist:/app/dist" charity-model
FROM python:3.12-slim
ENV KERAS_BACKEND=jax PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . && useradd -m app && chown -R app /app
USER app
ENTRYPOINT ["charity-model"]
CMD ["report", "--out", "dist", "--seed", "42"]
