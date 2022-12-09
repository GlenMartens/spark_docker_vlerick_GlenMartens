# Use an official Python runtime as a parent image
FROM python:3.9.13

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
