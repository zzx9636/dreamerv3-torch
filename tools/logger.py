import numpy as np
import wandb
import time

class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step
        wandb.init(project="dreamerv3", dir=str(logdir))

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        # input axes are (N, time, height, width, channel)
        # convert to axes are (time, channel, height, width, N)
        N, time, height, width, channel = value.shape
        value = np.array(value).transpose(1, 4, 2, 0, 3)
        # Stack the video to (time, channel, N*height, width) 
        value = np.reshape(value, (time, channel, height, width*N))
        
        self._videos[name] = value

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        wandb.log({"step": step, **dict(scalars)}, step=step)
        for name, value in self._images.items():
            wandb.log({name: wandb.Image(value)}, step=step)
        for name, value in self._videos.items():
            wandb.log({name: wandb.Video(value, fps=16, format="mp4")}, step=step)

        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        wandb.log({name: value}, step=step)

    def offline_video(self, name, value, step):
        wandb.log({name: wandb.Video(value, fps=16, format="mp4")}, step=step)