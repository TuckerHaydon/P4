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
    liblapack-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libsuitesparse-dev

# Configure
RUN git config --global user.name 'Snail Mail' && \
    git config --global user.email '<>'

# Install project
RUN mkdir -p /workspace/
RUN cd /workspace && \
    git clone https://github.com/tuckerhaydon/P4.git && \
    cd P4 && \
    git checkout gradient && \
    git submodule update --init --recursive
ENV P4=/workspace/P4
RUN cd $P4/dependencies/eigen && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make -j4 install
RUN cd $P4/dependencies/osqp && \
    mkdir build && \
    cd build && \
    cmake -G "Unix Makefiles" .. && \
    cmake --build . && \
    make -j4 install
RUN cd $P4/dependencies/ceres && \
    # Fix compile problem for c++11
    git cherry-pick e809cf0 && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
    # Configure for fast docker build. Not necessary.
    # The SCHUR option builds much faster, but runs slower
    -DCXX11=ON -DCXX11_THREADS=ON -DSCHUR_SPECIALIZATIONS=OFF .. && \
    make -j4 install
RUN mkdir -p $P4/build && \
    cd $P4/build && \
    cmake .. && \
    make -j4

# Log in directly to binaries
WORKDIR $P4/build/examples

# Startup
CMD ["/bin/bash"]
