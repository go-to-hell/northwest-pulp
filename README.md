BACKEND PROYERCTO DE ALGORITMOS
-----
# Intrucciones de instalación

Para correr este proyecto de FastAPI, primero necesitas instalar los requerimientos listados en el archivo requirements.txt. Aquí están los pasos que debes seguir:

1. Abre la terminal en tu IDE (Visual Studio Code).
2. Una vez estés en el directorio correcto, instala los requerimientos utilizando el comando pip install -r requirements.txt.
3. Después de que todas las dependencias estén instaladas, puedes correr el proyecto FastAPI con el comando uvicorn

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Una vez que lo tengas puedes entrar al endpoint `localhost:8000/docs` para ver la API navegable. En esa ventana puedes utilizar el botón `Try it out` para probar alguna función