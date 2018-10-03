import gym
import numpy as np

env = gym.make("Pendulum-v0")
#env.monitor.start("/tmp/pendulum3", force=True)


def run_episode(env, w, test):
    obs = env.reset()
    total = 0
    for i in range(env.spec.timestep_limit):
        if test:
            env.render()
        obs2 = np.outer(obs, obs).ravel()
        obs3 = np.outer(obs, obs2).ravel()
        x = np.concatenate([[1.0], obs, obs2, obs3])
        yhat = np.dot(w, x)
        action = np.array([4.0 / (1.0 + np.exp(-yhat)) - 2.0])
        obs, reward, done, info = env.step(action)
        total += reward
        if done:
            break
    return total


(n,) = env.observation_space.shape
mu = np.zeros(1 + n + n**2 + n**3)
sigma = np.identity(1 + n + n**2 + n**3) * 10


for epoch in range(30):
    print("Epoch", epoch)

    ws = np.random.multivariate_normal(mu, sigma, 300)
    scores = [run_episode(env, w, False) for w in ws]
    best = np.argsort(scores)[-10:]
    mu = np.mean(ws[best], axis=0)
    sigma = np.cov(ws[best].T)
    print("Score", np.mean(scores))
while 1:
    score = run_episode(env, mu, True)
print("Weights", mu)
print("Test Score", score)
