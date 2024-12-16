import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib
matplotlib.use("TkAgg")  # Смена бэкенда перед любым импортом matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Параметры
env_id = "MountainCarContinuous-v0"
# n_envs = 16
# n_steps = 16
# normalize_kwargs = {'norm_obs': True, 'norm_reward': False}
#
#
# def make_env():
#     return gym.make(env_id)


# Создаем векторизованное окружение
# envs = DummyVecEnv([make_env for _ in range(n_envs)])
# Нормализация окружения
# envs = VecNormalize(envs, **normalize_kwargs)
#
# model = PPO(
#     policy='MlpPolicy',
#     env=envs,
#     ent_coef=0.0,
#     gae_lambda=0.98,
#     gamma=0.99,
#     n_steps=n_steps,
#     n_epochs=4,
#     batch_size=256,  # так как 16*16=256
#     verbose=1,
#     device='cuda'  # Используем GPU, если доступен
# )

# Обучаем модель
# model.learn(total_timesteps=1_000_000)

# Сохраняем модель
# model.save("ppo_mountaincar")

# Загружаем модель (если потребуется)
model = PPO.load("ppo_mountaincar")

eval_env = gym.make(env_id, render_mode="rgb_array")

obs, info = eval_env.reset()
frames = []
total_reward = 0

# Количество шагов для визуализации
max_steps = 1000

for step in range(max_steps):
    # Для использования модели:
    # Если у вас VecNormalize, тогда obs нужно нормализовать или использовать eval_env.step(action)
    # Но при обычном gym окружении просто:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    total_reward += reward

    frame = eval_env.render()
    frames.append(frame)

    if done or truncated:
        print(f"Episode finished at step {step} with total reward: {total_reward}")
        break

# Создание анимации
fig, ax = plt.subplots()
ax.axis('off')
img = ax.imshow(frames[0])


def update(frame):
    img.set_data(frame)
    return [img]


ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=10)
plt.show()

eval_env.close()