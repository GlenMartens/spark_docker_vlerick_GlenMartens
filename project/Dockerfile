# Use an official Python runtime as a parent image
FROM datamechanics/spark:3.1-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3

WORKDIR /opt/application/

#RUN wget https://jdbc.postgresql.org/download/postgresql-42.2.5.jar
#RUN mv postgresql-42.2.5.jar /opt/spark/jars

COPY . /opt/application/

# Install any needed packages specified in requirements.txt
RUN pip3 install -r /opt/application/requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080
EXPOSE 4040

ENTRYPOINT ["python"]

CMD ["/opt/application/imdb.py"]
