# FROM public.ecr.aws/lambda/python:3.8

# # Copy function code
# COPY . ${LAMBDA_TASK_ROOT}

# # Install the function's dependencies using file requirements.txt
# # from your project folder.

# COPY requirements.txt  .
# RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# # Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
# CMD [ "app.handler" ] 

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

LABEL maintainer='nirisha@fusemachines.com'

# install as a package
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# copy code
COPY . /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]