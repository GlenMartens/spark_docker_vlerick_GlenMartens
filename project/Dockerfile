# Use an official Python runtime as a parent image
FROM datamechanics/spark:3.1-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3

WORKDIR /opt/application/

USER root
# Install apt packages and clean up cached files
#RUN apt-get update && \
#    apt-get install -y openjdk-11-jdk python3-venv && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

#RUN wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz && tar -xzf hadoop-3.3.1.tar.gz && mv hadoop-3.3.1 $HADOOP_HOME

RUN useradd -ms /bin/bash my_user
USER my_user

WORKDIR /repo

COPY . /repo/

# Install any needed packages specified in requirements.txt
RUN python -m pip install -r /repo/requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080
EXPOSE 4040

ENTRYPOINT ["python"]

CMD ["/repo/imdb.py"]
