# Use an official Python runtime as a parent image
FROM python:3.9.13

ENV PYSPARK_MAJOR_PYTHON_VERSION=3
ENV SPARK_LOCAL_IP=0.0.0.0

USER root
# Install apt packages and clean up cached files
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk 

#ENV JAVA_HOME /repo/java-11-openjdk/
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
RUN echo $JAVA_HOME

RUN useradd -ms /bin/bash python-api
USER python-api

WORKDIR /repo/

COPY . /repo/

# Install any needed packages specified in requirements.txt
RUN pip3 install -r /repo/requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080
EXPOSE 4040
EXPOSE 3000

ENTRYPOINT ["python"]

CMD ["/repo/imdb.py"]
