"""Custom Trainer callbacks for deep recurrent models."""

from transformers.trainer_callback import ProgressCallback


class QuietProgressCallback(ProgressCallback):
    """Keep tqdm progress bars while suppressing the default metric dict prints."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control
