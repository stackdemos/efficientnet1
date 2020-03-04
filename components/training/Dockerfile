FROM python:3.6
ARG EFFICIENTNET=git+git://github.com/SemanticMD/efficientnet_keras_transfer_learning
COPY requirements.txt /requirements.txt

WORKDIR /tmp
RUN apt-get update \
    && apt-get install -y --no-install-recommends python-pandas git \
    && pip3 install --no-cache-dir -U pip \
    && pip3 install --no-cache-dir -U -r /requirements.txt \
    && pip3 install --no-cache-dir -U ${EFFICIENTNET} \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    /usr/share/man /usr/share/doc /usr/share/doc-base /root/.cache

WORKDIR /src
COPY src .

