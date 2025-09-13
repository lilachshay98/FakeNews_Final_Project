# Fake News Detection Project README

This project includes several Python files for analyzing data across diverse datasets, provides the resulting figures, and 
defines the CredScore algorithm for fake news detection across the Internet and Twitter.

## Prerequisites

* PyCharm installed on your machine
* Git Large File Storage (LFS) installed  
  Git LFS is required to handle large datasets such as the Twitbot-20 dataset files.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development or testing purposes.

1. Clone the Repository
   
Clone this repository to your local machine using the following command:

  ```
  git clone https://github.com/lilachshay98/f5-assignment.git
  ```

2. Set up the Docker Images
   
The project already includes Docker images for the Nginx server and Test application in GitHub Container Registry. However, you can build them locally if needed:
  
  ```
  docker build -t ghcr.io/lilachshay98/nginx:2 ./nginx
  docker build -t ghcr.io/lilachshay98/test:2 ./test
  ```

3. Configuration Files

   * nginx.conf: The Nginx configuration file is located in the nginx/ directory. This file configures the Nginx server for proper operation.
   * requirements.txt: The requirements.txt file under the test/ directory contains the dependencies needed for the test application.
   * The test.py file is a Python script that sends HTTP requests to the Nginx server and verifies that it responds correctly. The test script is responsible for confirming that the Nginx server is functioning properly.


