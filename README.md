# xtream AI Challenge

## Ready Player 1? 🚀

Hey there! If you're reading this, you've already aced our first screening. Awesome job! 👏👏👏

Welcome to the next level of your journey towards the [xtream](https://xtreamers.io) AI squad. Here's your cool new assignment.

Take your time – you've got **10 days** to show us your magic, starting from when you get this. No rush, work at your pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### What You Need to Do

Think of this as a real-world project. Fork this repo and treat it as if you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the data and craft your own effective solutions.

🚨 **Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Keep This in Mind**: This isn't about building the fanciest model: we're more interested in your process and thinking.

---

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Meet Don Francesco, the mystery-shrouded, fabulously wealthy owner of a jewelry empire. 

He's got an impressive collection of 5000 diamonds and a temperament to match - so let's keep him smiling, shall we? 
In our dataset, you'll find all the glittery details of these gems, from size to sparkle, along with their values 
appraised by an expert. You can assume that the expert's valuations are in line with the real market value of the stones.

#### Challenge 1

Plot twist! The expert who priced these gems has now vanished. 
Francesco needs you to be the new diamond evaluator. 
He's looking for a **model that predicts a gem's worth based on its characteristics**. 
And, because Francesco's clientele is as demanding as he is, he wants the why behind every price tag. 

Create a Jupyter notebook where you develop and evaluate your model.

#### Challenge 2

Good news! Francesco is impressed with the performance of your model. 
Now, he's ready to hire a new expert and expand his diamond database. 

**Develop an automated pipeline** that trains your model with fresh data, 
keeping it as sharp as the diamonds it assesses.

#### Challenge 3

Finally, Francesco wants to bring your brilliance to his business's fingertips. 

**Build a REST API** to integrate your model into a web app, 
making it a cinch for his team to use. 
Keep it developer-friendly – after all, not everyone speaks 'data scientist'!

#### Challenge 4

Your model is doing great, and Francesco wants to make even more money.

The next step is exposing the model to other businesses, but this calls for an upgrade in the training and serving infrastructure.
Using your favorite cloud provider, either AWS, GCP, or Azure, design cloud-based training and serving pipelines.
You should not implement the solution, but you should provide a **detailed explanation** of the architecture and the services you would use, motivating your choices.

So, ready to add some sparkle to this challenge? Let's make these diamonds shine! 🌟💎✨

---

## How to run

### Install dependencies
First you need to create a new virtual environment and install the dependencies. You can do this by running the following commands with conda:

```bash
conda create -n env_name python=3.11
conda activate env_name
pip install -r requirements.txt
```

This env will be used to run all the code in this repository.

### Run training pipeline (Challenge 2)

The training pipeline has multiple arguments that can be passed to it. You can see all the arguments by running the following command:

```bash
python challenge_2/train.py --help
```

This will display the information needed to run the training pipeline. 

* `--dataset` - Path to the dataset file 
* `--output` - Path to export the trained model
* `--test_size` - Size of the test set (default=0.3)
* `--seed` - Random seed to be used in multiple functions (default=42)

To run the training pipeline you can use the following command:

```bash
python challenge_2/train.py --dataset datasets/diamonds/diamonds.csv
```

### Run REST API (Challenge 3)

To run the REST API we have 2 options. The first one is to run the API locally and the second one is to run the API using docker.

#### Environment variables
Before running the API you need to set the environment variables. You can do this by creating a `.env` file inside the `challenge_3` folder with the following content:

```bash
#Path where the model is saved
MODEL_PATH = '/app/model/my_model.json'

#Level of logging
PROD=INFO

#Port to run the API
PORT=8080
```

#### Local
Assuming that we are using the same environment created in the first step and that .env file was created, you can run the following command to start the API:

```bash
python challenge_3/app/main.py
```

#### Docker
To run the API using docker you need to build the image and run the container. You can do this by running the following commands:

```bash
docker compose build --up -d
```

This command is going to allow you to run the API in a container.

In this case, the `MODEL_PATH` should be set to `/app/model/my_model.json` because the model is going to be saved inside the container.


#### Testing the API

This API has a single endpoint that can be used to make predictions. The endpoint `/price` is a POST request that receives a JSON with the diamond features and returns the predicted price.

You can test the API by running the following command:

```bash
curl -X 'POST' \
  'http://localhost:8080/price' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "carat": 1.5,
  "cut": "Ideal",
  "color": "H",
  "clarity": "SI2",
  "depth": 62.0,
  "table": 55.0,
  "x": 6.63,
  "y": 6.65,
  "z": 4.10
}'
```

### Cloud infrastructure (Challenge 4)

The architecture for deploying this regression model will be centralized in the use of Amazon AWS tools. In this case, we will use Amazon API Gateway to manage the processing of API calls. This service is optimal because as it will allow us to manage traffic, access and perform continuous monitoring of the API.

The second service that we would need to use in this architecture would be Lambda Function as it allows us to connect to the service to be consumed, i.e. the Docker container of the API that performs the prediction of the price of diamonds. It should be clarified that such Docker image should be created and instantiated in Amazon Elastic Container Registry in order to use it.

![AWS Architecture](/challenge_4/CloudXtream.png)

A simplified version of this architecture could be a simple Lambda Function responsible for performing the inference of the trained model, in which case the model could be stored in S3 and then accessed by a script similar to the one developed in Challenge 3.