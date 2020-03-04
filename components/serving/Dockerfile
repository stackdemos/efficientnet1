FROM seldonio/seldon-core-s2i-python3

FROM ml1-harbor.svc.ml1.demo51.superhub.io/library/training:v24
RUN pip3 install --no-cache-dir -U 'seldon-core'

COPY --from=0 /microservice /microservice
COPY src/serving.py /microservice/.py

WORKDIR /microservice
ENTRYPOINT ["python","-u","microservice.py"]
CMD ["", "REST"]