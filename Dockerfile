FROM nvcr.io/nvidia/pytorch:25.02-py3
ARG RDKIT_VERSION=2026.3.5
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /work
COPY requirements-common.txt /tmp/requirements-common.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-common.txt rdkit==${RDKIT_VERSION}
COPY . /work
RUN python -m pip install --no-deps -e .
CMD ["python", "reproduce.py", "--help"]
