FROM pytorch/torchserve:latest

EXPOSE 8505
EXPOSE 8506
EXPOSE 8507

RUN pip3 install scikit-learn==1.1.2

COPY ./har.mar /home/model-server/model-store/
COPY ./config.properties /home/model-server/model-store/config.properties
ENV TS_CONFIG_FILE /home/model-server/model-store/config.properties
CMD ["torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models",  "har=har.mar","--ts-config","/home/model-server/model-store/config.properties"]
