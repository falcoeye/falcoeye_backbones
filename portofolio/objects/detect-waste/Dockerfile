FROM pytorch/torchserve:latest

EXPOSE 8500-9000

RUN pip install timm>=0.3.0
RUN pip install pycocotools>=2.0.2
RUN pip install omegaconf>=2.0



COPY ./detectwaste.mar /home/model-server/model-store/
COPY ./config.properties /home/model-server/model-store/config.properties
ENV TS_CONFIG_FILE /home/model-server/model-store/config.properties
CMD ["torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models",  "detectwaste=detectwaste.mar","--ts-config","/home/model-server/model-store/config.properties"]
