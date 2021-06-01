# CPPCNN

Convolutional neural network implementation using Facebook's recently released machine learning library [flashlight](https://github.com/flashlight/flashlight).

## Prerequisites
To avoid installing a ton of dependencies and later going through documentation pages if something goes wrong, I have containerized the entire application. So you will only need **docker**.


## Building Docker Image

To build the docker image, use
```bash
docker build -t <name>:<tag> .
```
where *name* is the name your want to give to the image and *tag* is a version number.

## Running The Docker Container
You can launch the docker container and mount the current working directory by
```bash
docker run -it -v "$(pwd)":/app <name>:<tag>
```

## Compiling
When inside the container, use the following command to build the application

> mkdir build && cd build

> cmake .. -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_PREFIX_PATH=/opt/onednn/lib/cmake/dnnl

> make -j$(nproc)

## Dataset
You can obtain the dataset used from [here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## License
[GPL3](https://www.gnu.org/licenses/gpl-3.0.en.html)