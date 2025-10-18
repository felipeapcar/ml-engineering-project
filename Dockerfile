# Imagen base de Python
FROM python:3.11-slim

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar requirements primero (mejora cache docker)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Comando por defecto (podemos actualizar esto luego)
CMD ["python", "scripts/run_pipeline.py"]