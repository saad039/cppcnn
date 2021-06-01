FROM flml/flashlight:cpu-latest AS flashlight
RUN apt update && apt install curl zip unzip tar

RUN git clone https://github.com/Microsoft/vcpkg.git

RUN cd vcpkg

RUN ./vcpkg/bootstrap-vcpkg.sh

RUN ./vcpkg/vcpkg install opencv

WORKDIR /app

COPY CMakeLists.txt .
COPY main.cpp .

