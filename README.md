# PyTorch Retro Game
## _Co-operation among bots_

This project was to show that communication with ML learning models speeds up the procces in
propertion to the number of agents. If there are _n_ agents and the model takes _k_ time to succeed
in a task then it will take _k/n_ time if there is communication involved among the agents.

## Features

- PyTorch Tensors for the Model
- Deep Q-Learning Model
- Gymnasium environments for learning: "knights_archers_zombies_v10"
- Pettingzoo for handeling multi-agent environments

The program runs 100,000 games with it gathering data and comparing the agents with and
without communication. After it will produce a csv with all the data so that we may create
a graph showing the growth over time. With the data collected over weeks of preparation 
I can see that I was correct with the _k/n_ time.

## Installation

To run this code you must start with creating a python3 environment.
```sh
python -m venv /path/to/new/virtual/environment
```
We then active the environment.
```sh
source env/bin/activate
```
Then we install all required libraries. I would just look into the files and start downloading
one-by-one.
```sh
pip install etc...
```