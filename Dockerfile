FROM python:3.12

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install uv
RUN pip install uv

RUN uv init 
RUN uv add streamlit pandas networkx gurobipy highspy pyarrow pydantic requests plotly

EXPOSE 8501

COPY ui.py .
COPY src/ src/

CMD ["uv", "run", "streamlit", "run", "--server.address=0.0.0.0", "--server.port=8501", "ui.py"]