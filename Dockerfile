FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Krasnoyarsk /etc/localtime \
    && echo "Asia/Krasnoyarsk" > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get install -y \
       git \
       ffmpeg \
       libsm6 \
       libxext6 \
       libgl1-mesa-glx \
       wget \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-библиотек:
# - ultralytics (указанная версия 8.3.38)
# - jupyterlab (для работы с ноутбуками)
# - другие зависимости
RUN pip install --no-cache-dir \
    ultralytics==8.3.38 \
    opencv-python-headless \
    numpy \
    scikit-learn \
    pyyaml \
    jupyterlab

# Переменная окружения для OpenCV
ENV QT_X11_NO_MITSHM=1

# Создадим рабочую директорию и скопируем проект
WORKDIR /app
COPY . /app

# Проброс порта для Jupyter Lab
EXPOSE 8888

# При запуске контейнера автоматически стартует Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
