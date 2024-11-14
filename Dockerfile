FROM neo4j:latest

RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install neo4j graphdatascience torch

EXPOSE 7474 7687
ENV NEO4J_AUTH=none
ENV NEO4J_PLUGINS='["graph-data-science","apoc"]'

# Set memory configurations
ENV NEO4J_dbms_memory_heap_initial__size=4G
ENV NEO4J_dbms_memory_heap_max__size=6G
ENV NEO4J_dbms_memory_pagecache_size=4G
ENV dbms.security.procedures.unrestricted=apoc.*


RUN echo 'user:x:${UID:-1000}:${GID:-1000}::/workspace:/bin/bash' >> /etc/passwd
