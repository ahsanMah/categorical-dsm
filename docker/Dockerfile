FROM nvcr.io/nvidia/pytorch:22.07-py3
ENV JUPYTER_TOKEN="niral"
ENV PASSWORD=niral

ARG USER=docker
ARG UID=1000
ARG GID=1000
RUN apt-get update --fix-missing && apt install htop

RUN useradd --system --create-home --home-dir /home/${USER} \
    --shell /bin/bash --gid root -G sudo --uid ${UID} ${USER}
RUN echo ${USER}:docker | chpasswd
USER ${USER}
WORKDIR /home/${USER}
ENV PATH="${PATH}:/home/${USER}/.local/bin"

# Repo requirements
RUN python -m pip install -U pip
COPY repo_requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
RUN python -m pip install -U jupyterlab jupyterlab_widgets ipywidgets
# ENTRYPOINT ["jupyter", "lab", "--port=8888", "--notebook-dir=/", "--no-browser", "--ip=0.0.0.0"]

# RUN apt install -y htop nodejs

# Add zsh ricing
USER root
RUN apt install -y zsh wget && chsh -s $(which zsh)
USER ${USER}

# Uses "Spaceship" theme with some customization. Uses some bundled plugins and installs some more from github
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

RUN conda config --set changeps1 False && conda config --set auto_activate_base False