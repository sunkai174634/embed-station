FROM embed-station-runtime:2026.3
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:${PYTHONPATH}
WORKDIR /app
USER root
RUN mkdir -p /data /cache/gpu && chown 1000:1000 /data /cache/gpu
COPY src ./src
USER openvino
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=3)"
CMD ["python3", "-m", "uvicorn", "embed_station.api:app", "--host", "0.0.0.0", "--port", "8080"]
