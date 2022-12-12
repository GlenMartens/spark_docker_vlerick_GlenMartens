# Use an official Python runtime as a parent image
FROM gitpod/workspace-python-3.8:2022-12-02-22-15-49

USER root

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk python3-venv

RUN useradd -ms /bin/bash my_user
USER my_user

WORKDIR /repo

COPY . /repo/

# Install any needed packages specified in requirements.txt
RUN python -m pip install -r /repo/requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

ENTRYPOINT ["python"]

CMD ["/repo/imdb.py"]
