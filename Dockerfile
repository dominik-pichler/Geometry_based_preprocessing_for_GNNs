FROM neo4j:latest

RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install neo4j graphdatascience torch prometheus_client

EXPOSE 7474 7687

ENV NEO4J_AUTH=none

RUN echo "user:x:${UID:-1000}:${GID:-1000}::/workspace:/bin/bash" >> /etc/passwd
