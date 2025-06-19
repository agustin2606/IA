import gymnasium as gym
import imageio
import numpy as np
import os
from datetime import datetime

class VideoRecorderWrapper(gym.Wrapper):
    def __init__(self, env, output_dir="videos", filename=None, record_every=1, fps=15):
        super().__init__(env)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.frames = []
        self.episode_count = 0
        self.record_every = record_every
        self.fps = fps
        self.filename = filename

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.frames = []
        if self.episode_count % self.record_every == 0:
            frame = self.env.render() 
            if frame is not None:
                self.frames.append(frame)
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if self.episode_count % self.record_every == 0:
            frame = self.env.render()
            if frame is not None:
                self.frames.append(frame)
        if done or truncated:
            if self.episode_count % self.record_every == 0:
                self._save_video()
            self.episode_count += 1
        return obs, reward, done, truncated, info

    def _save_video(self):
        try:
            max_width = max(img.shape[1] for img in self.frames)
            max_height = max(img.shape[0] for img in self.frames)

            # Rellenar con fondo blanco si alguna imagen es más chica
            normalized_frames = []
            for img in self.frames:
                h, w = img.shape[:2]
                new_img = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255  # blanco
                new_img[:h, :w, :] = img
                normalized_frames.append(new_img)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.filename:
                filename = os.path.join(self.output_dir, self.filename+".mp4")
            else:
                filename = os.path.join(self.output_dir, f"episode_{self.episode_count}_{timestamp}.mp4")
            print(f"[VideoRecorderWrapper] Saving video to {filename}")
            imageio.mimsave(filename, normalized_frames, fps=self.fps)
        except Exception as e:
            raise RuntimeError(f"Error saving video. Remember to set 'render_mode' to 'rgb_array' in env. Error {e}")


    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
