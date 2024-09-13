# base image
FROM tensorflow/tensorflow:latest-gpu

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update && apt-get install -y ffmpeg

# making directory of app
WORKDIR /streamlit-docker

# copy over requirements
COPY requirements.txt ./requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install pip then packages
RUN pip install --upgrade pip 
RUN pip install cython 
RUN pip install numpy==1.21
RUN pip install click==7.1.2 
RUN pip install -r requirements.txt

# copying all files over
COPY . .

# streamlit-specific commands for config
RUN mkdir -p /root/.streamlit \
    && bash -c 'echo -e "\
    [general]\n\
    email = \"\"\n\
    " > /root/.streamlit/credentials.toml' \
    && bash -c 'echo -e "\
    [server]\n\
    enableCORS = false\n\
    " > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

# cmd to launch app when container is run
CMD streamlit run app.py