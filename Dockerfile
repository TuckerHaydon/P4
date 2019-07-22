FROM ubuntu:18.04

LABEL "author"="tuckerhaydon"
LABEL "email"="tuckerhaydon@gmail.com"

# Install tools
RUN apt-get update && apt-get -y install --no-install-recommends \
    ca-certificates \
    git \
    g++ \
    make \
    cmake \
    libblas-dev \ 
    liblapack-dev

# Make workspace directory
RUN mkdir /workspace

# Install libraries
RUN mkdir -p /workspace/libs
WORKDIR /workspace/libs

RUN git clone https://github.com/eigenteam/eigen-git-mirror.git eigen && \
    cd eigen && \
    git checkout tags/3.3.7 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . && \
    make install

RUN git clone https://github.com/oxfordcontrol/osqp.git && \
    cd osqp && \
    git submodule update --init --recursive && \
    mkdir build && \
    cd build && \
    cmake -G "Unix Makefiles" .. && \
    make && \
    make install

# Install project
RUN mkdir -p /workspace/P4
WORKDIR /workspace/P4

COPY CMakeLists.txt /workspace/P4/
COPY src /workspace/P4/src/
COPY examples /workspace/P4/examples

RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    make

# Log in directly to binaries
WORKDIR /workspace/P4/build/examples

# Startup
CMD ["/bin/bash"]

