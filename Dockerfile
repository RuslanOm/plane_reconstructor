FROM pytorch/pytorch:latest
COPY . /plane_reconstructor
WORKDIR /plane_reconstructor

RUN pip install -r requirements.txt
EXPOSE 5000

ENTRYPOINT ["python", "main.py"]
