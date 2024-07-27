# ML Assignment 2: Report

## Introduction

### Background

In this report, we assume we are a team of machine learning engineers working for a technology company that is manufacturing routers. Our team is tasked with developing a reinforcement learning-based scheduling algorithm for a router currently being designed by our company. We will discuss our task, explain our approach, and present our methodology and results. We will also compare our ML policy against well-known pre-existing network scheduling policies.

## Task

The task involves using a scheduler to pop an item from any of the three provided queues. We calculate the mean delay for each queue, which refers to the average delay for all packets in the queue, measured in timeslots. The QoS constraints are outlined in the table below. Our goal is to keep the best effort queue to a minimum.

| Metric                | Priority Queue | Priority Queue 2 | Best Effort Queue |
|-----------------------|----------------|------------------|-------------------|
| Mean Delay Requirement| 6              | 4                | inf               |

Every timeslot, a certain amount of packets will enter each queue. The goal of the scheduler is to maintain the mean delay for queue 1 and queue 2 below the mean delay requirement, while minimizing the mean delay for the best effort queue. This will be performed under the following scenarios:

### Scenario 1
The scheduler per timeslot can either pop a packet off queue 1, queue 2, or queue 3, resulting in 3 possible actions.

### Scenario 2
The scheduler per timeslot can either pop from the current queue or switch to one of the other two queues, resulting in a defined action space of 3 as well.

## Approach

In OpenAI Gym, we set up an environment for visualizing the waiting time for each queue using Reinforcement Learning. The objective is to let an agent find an optimal policy over time through trial and error, iteratively mapping states to actions and maximizing the cumulative reward.

The Q-learning update formula used is:

```plaintext
Q(s, a) = Q(s, a) + learning_rate * (reward + gamma * max(Q(s', a')) - Q(s, a
```
## Actions Established

- **Scenario 1**: Popping packets from queue 1, 2, or 3.
- **Scenario 2**: Popping from the current queue or switching to another queue (counts as 2 actions, as there are 2 other queues).

The observation space is discrete with 1,000,000,000 possibilities to simplify the Q-learning process. After setting up the baseline model, we will adjust hyperparameters to find the optimal configuration.

## Implementation

### Baseline Model and Environment

The model will be trained in our custom OpenAI Gym environment over 1500 episodes, each with a maximum of 100 steps. The model will:

- Reward a +1 score if it improves the mean delay of the best effort queue.
- Penalize -1 score if it increases the size of the best effort queue.
- Apply a penalty of -10 for exceeding the mean delay requirement for queues 1 and 2.
- Reward +25 if all three queues are empty.
- Deduct -25 if it pops from an empty queue when packets are still in other queues.

Each episode will have randomized arrival rates for each queue, but the sum of the rates will be within the range of 0.9 to 1.5 to avoid unfair penalties. The model will be evaluated based on the average cumulative reward, and each episode will be plotted to determine policy convergence.

The baseline model has gamma and learning rates of 0.1. These will be further tuned, with the baseline results showing:
- **Scenario 1**: Average cumulative reward score of -492.765
- **Scenario 2**: Average cumulative reward score of -415.6773333333333

### Hyperparameter Tuning

Hyperparameters tuned include learning rate and gamma value:
- Gamma values range from 0.1 to 0.9 in increments of 0.1.
- Learning rates range from 0.1 to 0.5 in increments of 0.1, capped at 0.5 to avoid oscillation or overshooting issues.

A grid search method will test combinations for 400 episodes per permutation to reduce computational complexity. The model with the best average cumulative reward will be selected.

**Optimal hyperparameters found:**
- **Scenario 1**: Gamma: 0.6, Learning Rate: 0.5
- **Scenario 2**: Gamma: 0.2, Learning Rate: 0.5

### Testing Final Model

The final model was tested with an arrival rate of [0.3, 0.4, 0.4], which was not included in the training to ensure consistency. The model relies on its Q-table for different states rather than having been specifically trained on this arrival rate. This test was conducted over 50 episodes.

Results indicate that Scenario 1 performed better with Q-learning than Scenario 2.

### Independent Evaluation

Our ML model was compared with other popular scheduling policies such as FIFO, EDF, and SP:
- **FIFO** performed the worst.
- **EDF** and **SP** performed similarly to ML.

With arrival rates set to 0.3, 0.4, and 0.4, lower rates did not differentiate between models.

Given the manageable queue size and packet transmission, EDF and SP are simpler and more effective. However, in more complex scenarios with additional queues and delay requirements, the ML policy could prove more effective.

## Conclusion

Our results show that we were able to minimize the best effort queue while managing priority queues. The ML policy scored 147, close to SP and EDF's score of 148. However, our model struggles with arrival rates above 1.5. Future work should involve increasing training rates and data, and exploring more complex queue scenarios. The best approach would be to train the model according to real-world maximum packet arrival rates.

## References

- **Paperspace Blog**: [Getting Started with OpenAI Gym](https://blog.paperspace.com/getting-started-with-openai-gym/)
- **Guo, X. (2018, July 24)**: [Reinforcement Learning with OpenAI](https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2)
- **Nicholas Renotte (2020, October 5)**: [OpenAI Gym Tutorial - Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=bD6V3rcr_54&ab_channel=NicholasRenotte)
