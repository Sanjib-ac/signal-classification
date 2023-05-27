-------------------------------------------------------------------------------------------------------------------------------------
 Panoradio HF Dataset
 Dataset for RF signal classification
 Stefan Scholl, 2019

 Corresponding Publication:
 S. Scholl, "Classification of Radio Signals and HF Transmission Modes with Deep Learning", June 2019, https://arxiv.org/abs/1906.04459
-------------------------------------------------------------------------------------------------------------------------------------

Data stored as 2-D numpy array with shape=(172800, 2048)
172800 signals
2048 IQ samples (complex)
fs = 6000 Hz
power normalized to 1
frequency offset +- 250 Hz
SNR: 25, 20, 15, 10, 5, 0, -5, -10 dB
fading channels: CCIR 520
modes:
"morse", 
"psk31",
"psk63",
"qpsk31",
"rtty45_170",
"rtty50_170",
"rtty100_850",
"olivia8_250",
"olivia16_500",
"olivia16_1000",
"olivia32_1000",
"dominoex11",
"mt63_1000",
"navtex",
"usb",
"lsb",
"am",
"fax"