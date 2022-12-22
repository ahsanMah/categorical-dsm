#!/bin/bash

docker build ./ --build-arg USER=$USER \
				--build-arg UID=$(id -u) \
				-t ahsanmah/pytorch_sde