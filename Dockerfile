FROM nvcr.io/nvidia/tensorflow:21.02-tf1-py3
ENV DEBIAN_FRONTEND noninteractive
RUN pip install dlib
RUN apt-get update -y
RUN pip install albumentations==0.5.2
RUN pip install blend-modes==2.1.0
RUN pip install dominate==2.6.0
RUN pip install numpy==1.17.3
RUN pip install opencv-python==4.6.0.66
RUN pip install Pillow
RUN pip install pretrainedmodels==0.7.4
RUN pip install scikit-image
RUN pip install segmentation-models-pytorch==0.1.3
RUN pip install tensorboardX==2.2
RUN pip install tqdm==4.59.0
RUN pip install timm==0.4.12
RUN wget https://public.vinai.io/CPM_checkpoints/color.pth
RUN wget https://public.vinai.io/CPM_checkpoints/pattern.pth
RUN apt install libgl1-mesa-glx -y
