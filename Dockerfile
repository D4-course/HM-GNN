FROM continuumio/anaconda3

WORKDIR /src/
COPY . .
RUN apt-get update && apt-get install -y gcc libstdc++6
RUN cp /opt/conda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/
RUN cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
RUN conda env create --file environment.yml

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]