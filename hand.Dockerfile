FROM continuumio/miniconda3

COPY . /

SHELL ["/bin/bash", "-l", "-c"]
RUN conda env create -f /environment.yml && \
    conda clean -afy && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.profile && \
    echo "conda activate asf-tools" >> /root/.profile && \
    conda activate asf-tools

RUN python3 -m pip install -e .

ENTRYPOINT ["/entry.sh"]