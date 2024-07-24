FROM python:3.9-slim

# Configura el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios
COPY pyproject.toml poetry.lock /app/

# Instala Poetry
RUN pip install poetry

# Configura Poetry para usar la versión de Python correcta
RUN poetry config virtualenvs.create false

# Instala las dependencias
RUN poetry install --no-dev

# Copia el resto del código
COPY . /app/

# Define el comando para ejecutar tu aplicación
CMD ["python", "main.py"]

