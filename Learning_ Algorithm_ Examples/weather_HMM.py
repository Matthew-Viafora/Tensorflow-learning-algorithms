#CREDIT: https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=DJ0XIA2M5gqD
#Cold days are encoded by a 0 and hot days are encoded by a 1
#The first day in our squence has an 80% chance of being cold
#A cold day has a 30% chance of being followed by a hot day
#A hot day has a 20% chacne of being followed by a cold day
#On each day the termperature is normally distributed with mean and standard deviation 0 and 6 on a cold day and mean and standard deviation 15 and 10 on a hot day
import tensorflow_probability as tfp
import tensorflow as tf

#making a shortcut for later on
tfd = tfp.distributions
#refer to point 2 in description above (80% cold, 20% hot on first day)
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
#refer to point 3 in description above (70% cold, 30% hot on transition)
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                 [0.2, 0.8]]) #refer to point 4 in description above (20% cold and 80% hot on transition)
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.]) #refer to point 5 above
#loc represents mean and scale represents standard deviation

model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 100
)

mean = model.mean()
#evaluate model within a session and get expected temperatures on each day
with tf.compat.v1.Session() as sess:
    print(mean.numpy())