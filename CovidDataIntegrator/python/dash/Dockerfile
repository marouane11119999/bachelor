FROM python:3.9

WORKDIR /usr/app/src


RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH


COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords -d /usr/local/share/nltk_data && \
    python -m nltk.downloader punkt -d /usr/local/share/nltk_data && \
    python -m nltk.downloader punkt_tab -d /usr/local/share/nltk_data

RUN mkdir -p /usr/app/src/uploaded_files

COPY app.py ./app.py
COPY pages /usr/app/src/pages
COPY assets /usr/app/src/assets
COPY helpers /usr/app/src/helpers

CMD ["python", "app.py"]
