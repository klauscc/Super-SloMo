"""
single frame interpolation.
"""
from time import time
# import click
import cv2
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import model
from torchvision import transforms
from torch.functional import F
import glob, os
import argparse

import ode_unet

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="path to model checkpoint")

args = parser.parse_args()

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()
if device != "cpu":
    mean = [0.429, 0.431, 0.397]
    mea0 = [-m for m in mean]
    std = [1] * 3
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std), trans_backward])


# The workaround if the model is trained with nn.DataParallel.
class WrappedModel(nn.Module):

    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module    # that I actually define.

    def forward(self, *args):
        return self.module(*args)


# flow = model.UNet(6, 4).to(device)
ode_method = "euler"
ode_step_size = 0.0625
flow = WrappedModel(ode_unet.ODE_UNet(6, 2, ode_method, ode_step_size)).to(device)
interp = WrappedModel(model.UNet(20, 5)).to(device)
back_warp = None


def setup_back_warp(w, h):
    global back_warp
    with torch.set_grad_enabled(False):
        back_warp = model.backWarp(w, h, device).to(device)


def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')
    interp.load_state_dict(states['state_dictAT'])
    flow.load_state_dict(states['state_dictFC'])


def interpolate_batch(frames, factor=2):
    frame0 = torch.stack(frames[:-1])
    frame1 = torch.stack(frames[1:])

    i0 = frame0.to(device)
    i1 = frame1.to(device)
    ix = torch.cat([i0, i1], dim=1)

    # flow_out = flow(ix)
    # f01 = flow_out[:, :2, :, :]
    # f10 = flow_out[:, 2:, :, :]
    inter_ts = torch.tensor(np.array([0.5])).to(device)
    ft0, f10 = flow(inter_ts, torch.cat((i0, i1), dim=1))
    ft1, f01 = flow(inter_ts, torch.cat((i1, i0), dim=1))

    frame_buffer = []
    for i in range(1, factor):
        t = i / factor
        # temp = -t * (1 - t)
        # co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

        # ft0 = co_eff[0] * f01 + co_eff[1] * f10
        # ft1 = co_eff[2] * f01 + co_eff[3] * f10

        gi0ft0 = back_warp(i0, ft0)
        gi1ft1 = back_warp(i1, ft1)

        iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
        io = interp(iy)

        ft0f = io[:, :2, :, :] + ft0
        ft1f = io[:, 2:4, :, :] + ft1
        vt0 = F.sigmoid(io[:, 4:5, :, :])
        vt1 = 1 - vt0

        gi0ft0f = back_warp(i0, ft0f)
        gi1ft1f = back_warp(i1, ft1f)

        co_eff = [1 - t, t]

        ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
               (co_eff[0] * vt0 + co_eff[1] * vt1)

        frame_buffer.append(ft_p)

    return frame_buffer


def load_batch(video_in, batch_size, batch, w, h):
    if len(batch) > 0:
        batch = [batch[-1]]

    for i in range(batch_size):
        ok, frame = video_in.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((w, h), Image.ANTIALIAS)
        frame = frame.convert('RGB')
        frame = trans_forward(frame)
        batch.append(frame)

    return batch


def load_image(img_path):
    frame = cv2.imread(img_path, 1)
    h0, w0, _ = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # print(frame.shape)
    # (1080, 1920, 3)
    frame = Image.fromarray(frame)

    w, h = (w0 // 32) * 32, (h0 // 32) * 32
    setup_back_warp(w, h)
    frame = frame.resize((w, h), Image.ANTIALIAS)
    frame = frame.convert('RGB')
    frame = trans_forward(frame)
    # batch.append(frame)

    return frame, h0, w0


def denorm_frame(frame, w0, h0):
    frame = frame.cpu()
    frame = trans_backward(frame)
    frame = frame.resize((w0, h0), Image.BILINEAR)
    frame = frame.convert('RGB')
    return np.array(frame)[:, :, ::-1].copy()


def convert_video(source, dest, factor, batch_size=10, output_format='mp4v', output_fps=30):
    vin = cv2.VideoCapture(source)
    count = vin.get(cv2.CAP_PROP_FRAME_COUNT)
    w0, h0 = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codec = cv2.VideoWriter_fourcc(*output_format)
    vout = cv2.VideoWriter(dest, codec, float(output_fps), (w0, h0))

    w, h = (w0 // 32) * 32, (h0 // 32) * 32
    setup_back_warp(w, h)

    done = 0
    batch = []
    while True:
        batch = load_batch(vin, batch_size, batch, w, h)
        if len(batch) == 1:
            break
        done += len(batch) - 1

        intermediate_frames = interpolate_batch(batch, factor)
        intermediate_frames = list(zip(*intermediate_frames))

        for fid, iframe in enumerate(intermediate_frames):
            vout.write(denorm_frame(batch[fid], w0, h0))
            for frm in iframe:
                vout.write(denorm_frame(frm, w0, h0))

        try:
            yield len(batch), done, count
        except StopIteration:
            break

    vout.write(denorm_frame(batch[0], w0, h0))

    vin.release()
    vout.release()


def interpolate_singleframe():

    frame1s = sorted(glob.glob('../UCF101_results/ucf101_interp_ours/*/*00.png'))
    frame2s = sorted(glob.glob('../UCF101_results/ucf101_interp_ours/*/*02.png'))

    for frame1_path, frame2_path in zip(frame1s, frame2s):
        assert frame1_path[:-6] == frame2_path[:-6]
        framename = frame1_path.split('/')

        frame1, h0, w0 = load_image(frame1_path)
        frame2, h0, w0 = load_image(frame2_path)

        batch = [frame1, frame2]

        intermediate_frames = interpolate_batch(batch, 2)
        # intermediate_frames contains only one frame!
        for int_frame in intermediate_frames:
            frame = int_frame.cpu()[0]
            # frame = transtoimage(frame)
            frame = trans_backward(frame)

            frame = frame.convert('RGB').resize((w0, h0), Image.BICUBIC)
            frame.save(
                os.path.join('../UCF101_results/ucf101_interp_ours/%s/frame_01_ode.png' %
                             (framename[-2])), 'PNG')


load_models(args.checkpoint)
interpolate_singleframe()
