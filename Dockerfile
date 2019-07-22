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

# Install project
RUN mkdir -p /workspace/
RUN cd /workspace && \
    git clone https://github.com/tuckerhaydon/P4.git && \
    cd P4 && \
    git submodule update --init --recursive
ENV P4=/workspace/P4
RUN cd $P4/dependencies/eigen && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install
RUN cd $P4/dependencies/osqp && \
    mkdir build && \
    cd build && \
    cmake -G "Unix Makefiles" .. && \
    cmake --build . && \
    make install
RUN mkdir -p $P4/build && \
    cd $P4/build && \
    cmake .. && \
    make

# Log in directly to binaries
WORKDIR $P4/build/examples

# Startup
CMD ["/bin/bash"]
