FROM ml1-harbor.svc.ml1.demo51.superhub.io/library/training:v31
RUN pip3 install --no-cache-dir -U 'flask>=0.12.3' \
    && pip3 install --no-cache-dir -U 'kfp==0.1.23' \
    && pip3 install --no-cache-dir -U 'boto3>=1.0'
COPY src/ /app
WORKDIR /app
ENTRYPOINT ["python","-u","app.py"]