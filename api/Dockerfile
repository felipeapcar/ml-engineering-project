FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements para la API
COPY api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la API
COPY api/ /app/api
COPY models/ /app/models/

# Exponer puerto para FastAPI
EXPOSE 8080

# Comando para correr FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]