FROM public.ecr.aws/lambda/python:3.8

ARG MODEL_DIR=./models

ENV SENTENCE_TRANSFORMERS_HOME=$MODEL_DIR
ENV TRANSFORMERS_CACHE=$MODEL_DIR
ENV TRANSFORMERS_VERBOSITY=error

RUN yum -y install gcc-c++

# Copy function code
COPY . ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN chmod -R 0777 $MODEL_DIR

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ] 

# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# LABEL maintainer='nirisha@fusemachines.com'

# # install as a package
# COPY requirements.txt /app/requirements.txt
# RUN pip install -r requirements.txt

# # copy code
# COPY . /app

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]