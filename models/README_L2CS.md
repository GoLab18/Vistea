# L2CS-Net gaze weights

The appearance-based (CNN) calibration uses **L2CS-Net**, a ResNet-50
trained on the **Gaze360** dataset, to regress two gaze angles (pitch, yaw)
from a face crop. The pretrained weights are optional: without them
the geometric strategies keep working, and the L2CS strategy simply reports
that it is unavailable.

## Expected file

Place the checkpoint here:

    models/L2CSNet_gaze360.pkl

It is a pickled PyTorch `state_dict` containing a ResNet-50 backbone and two 90-bin angular classification heads (`fc_pitch_gaze`, `fc_yaw_gaze`).
An unused `fc_finetune` head present in the checkpoint is safely ignored (the state dict is loaded non-strictly).

## How to obtain it

Official method and weights: **Ahmednull/L2CS-Net** on GitHub (<https://github.com/Ahmednull/L2CS-Net>).
To download the **Gaze360** weights file (`L2CSNet_gaze360.pkl`) refer to its documentation.

A public mirror of the same file is also hosted on the Hugging Face Hub, e.g.:

    https://huggingface.co/tianfxc/l2cs/resolve/main/L2CSNet_gaze360.pkl

Download it and move it to `models/L2CSNet_gaze360.pkl`.

## Runtime extras

The network needs the optional `cnn` dependency group (PyTorch + torchvision):

    uv sync --group cnn

`core.appearance.available()` returns `True` only when both torch is
importable **and** the weights file is present.