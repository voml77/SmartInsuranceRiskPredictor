FROM public.ecr.aws/lambda/python:3.9
# Arbeitsverzeichnis im Container
WORKDIR /var/task
# Kopiere alle Dateien in das Container-Verzeichnis
COPY . .

# Standardbefehl für den Lambda-Runtime-Start
CMD ["lambda_function.lambda_handler"]
