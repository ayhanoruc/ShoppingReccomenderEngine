# Python base image
FROM python:3.11-slim


# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends


# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN pip install -r requirements.txt

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

# Define the command to run the application using uvicorn # 80:8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 
