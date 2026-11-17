# Image for the Kubernetes side: the replica proxy, and (local test) the fake GPU machines.
#   docker build -t vpa-k8s:dev -f poc/k8s-keda/Dockerfile .
FROM python:3.12-slim
RUN pip install --no-cache-dir aiohttp==3.11.18 prometheus-client==0.21.1 PyYAML==6.0.2
WORKDIR /app
COPY controller/precision_controller.py /app/controller/
COPY poc/k8s-keda/agent.py poc/k8s-keda/fake_vllm.py poc/k8s-keda/proxy.py poc/k8s-keda/controller.yaml /app/poc/k8s-keda/
ENV PYTHONUNBUFFERED=1
