import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib

matplotlib.use("TkAgg")


def animate(model, save=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    eval_env = gym.make(ENV_ID, render_mode="rgb_array")

    obs, _ = eval_env.reset()
    frames = []
    total_reward = 0

    max_steps = 1000
    for step in range(max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward

        frame = eval_env.render()
        frames.append(frame)

        if done or truncated:
            print(f'{total_reward = }')
            break

    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return [img]

    ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=10)

    if save:
        ani.save("animation.mp4", writer=FFMpegWriter(fps=60))

    plt.show()

    eval_env.close()


def learn(save=False):
    n_envs = 16
    n_steps = 64
    normalize_kwargs = {'norm_obs': True, 'norm_reward': False}
    policy_kwargs = dict(net_arch=[64, 64])

    envs = DummyVecEnv([lambda: gym.make(ENV_ID) for _ in range(n_envs)])
    envs = VecNormalize(envs, **normalize_kwargs)

    model = PPO(
        policy='MlpPolicy',
        env=envs,
        policy_kwargs=policy_kwargs,
        ent_coef=0.001,
        gae_lambda=0.98,
        gamma=0.99,
        n_steps=n_steps,
        n_epochs=10,
        batch_size=n_envs*n_steps,
        tensorboard_log="./ppo_mountaincar_tensorboard/",
        verbose=1,
        device='cuda'
    )

    model.learn(total_timesteps=500_000)
    if save:
        model.save("ppo_mountaincar")

    return model


LEARN = False
ENV_ID = "MountainCarContinuous-v0"


if __name__ == '__main__':
    if LEARN:
        model = learn(save=True)
    else:
        model = PPO.load("ppo_mountaincar")

    animate(model, save=True)
