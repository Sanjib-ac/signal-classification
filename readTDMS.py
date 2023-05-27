import time

from matplotlib.animation import FuncAnimation
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
import PyQt5


def read_tdms(file_name):
    with TdmsFile.open(fileName) as tdms_file:
        all_groups = tdms_file.groups()
        print(f'all_groups: {all_groups}')
        all_group_channels = all_groups[0].channels()
        # print(all_group_channels)
        raw_data = all_group_channels[0][:]
        # print(len(all_channel_data))
    return raw_data


if __name__ == "__main__":
    # loading the npy data
    # OPen a TDMS file
    # fileName = 'Test Streaming 2.tdms'
    fileName = 'demo_file.tdms'
    all_channel_data = read_tdms(fileName)
    # print(len(all_channel_data))
    # pltdata = all_channel_data[0:4098]
    # x = [i for i in range(2049)]
    # y = pltdata[0::2]
    # z = pltdata[1::2]
    # plt.plot(x, y)
    # plt.plot(x, z)
    # print(len(y), len(z), len(x))
    # plt.show()

    modulation_classes = ["morse", "psk31", "psk63", "qpsk31", "rtty45_170",
                          "rtty50_170", "rtty100_850", "olivia8_250", "olivia16_500",
                          "olivia16_1000", "olivia32_1000", "dominoex11", "mt63_1000",
                          "navtex", "usb", "lsb", "am", "fax"]
    morse_last_idx = 9599 * 2048 * 2

    mod_with_data = {}
    start = 0
    step = (9600 * 2048) * 2
    for i in range(len(modulation_classes)):
        mod_with_data[modulation_classes[i]] = all_channel_data[start: start + step]
        start = start + step
    # print(mod_with_data)
    key_to_value_lengths = {k: len(v) for k, v in mod_with_data.items()}
    # print(f'length of the values', key_to_value_lengths)
    len_of_mod = len(mod_with_data['morse'])

    t = [i for i in range(int(len_of_mod / 2))]
    s_idx = 0
    fig, ax = plt.subplots()
    # plt.plot(mod_with_data['psk31'][0::2][s_idx:s_idx + 2048])
    # plt.plot(mod_with_data['psk31'][0::2][0:5000])
    # plt.plot(mod_with_data['psk31'][1::2][0:5000])
    # plt.plot(t, mod_with_data['morse'][1::2][0:5000])
    f, tme, sxx = sc.signal.spectrogram(mod_with_data['psk31'][0::2][0:5000], 6e3)

    plt.pcolor(tme, f, sxx)
    # plt.specgram(mod_with_data['psk31'][0::2][0:5000])
    ax.set_facecolor('Black')
    # fig, ax = plt.subplots()
    # ln, = ax.plot([], [], 'ro')
    # x = [i for i in range(2048)]
    #
    # def running_plot(i):
    #     ax.clear()
    #     y = all_morse_r[i:i + 2048]
    #     i += 2048
    #     ax.plot(x, y)
    #
    #
    # ani = FuncAnimation(fig, running_plot, frames=len(all_morse_r), interval=1)
    plt.tight_layout()
    plt.show()
