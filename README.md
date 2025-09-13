# Fake News Detection Project README

This project includes several Python files for analyzing data across diverse datasets, provides the resulting figures, and 
defines the CredScore algorithm for fake news detection across the Internet and Twitter.

## Prerequisites

* Pycharm installed on your machine

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development or testing purposes.
1. Clone the repository
   Clone this repository to your local machine using the following command:

  ```
  git clone https://github.com/lilachshay98/needle.git
  ```
2. Git LFS setup
   Since the Twitbot-20 datasets (dev.json, train.json, test.json) are very large, in order to use them it is required to perform the following:
  ```bash
  git lfs install
  git lfs pull          # fetch the real blobs for all LFS-tracked files
  git lfs checkout      # ensure working tree files are replaced (not pointer stubs)
