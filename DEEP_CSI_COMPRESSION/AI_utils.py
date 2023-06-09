import numpy as np
import matplotlib.pyplot as plt

def plot_loss(tr_avg, tr_upper, tr_lower, val_avg, val_upper, val_lower, xtick = None):
    x = np.arange(len(tr_avg))
    fig, ax = plt.subplots()
    ax.plot(tr_avg, color = 'b', label = 'train_loss')
    # ax.plot(val_avg, color = 'r', label = 'val_loss')
    ax.fill_between(x, tr_upper, tr_lower, color='b', alpha=.3)
    # ax.fill_between(x, val_upper, val_lower, color='r', alpha=.15)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_title('Model training and validation performance')
    plt.legend()
    
    if xtick != None:
        ax.set_xticklabels([0,1,10,20,30,40,50,60,70, 80])
    
    fig, ax = plt.subplots()
    # ax.plot(tr_avg, color = 'b', label = 'train_loss')
    ax.plot(val_avg, color = 'r', label = 'val_loss')
    # ax.fill_between(x, tr_upper, tr_lower, color='b', alpha=.15)
    ax.fill_between(x, val_upper, val_lower, color='r', alpha=.3)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_title('Model training and validation performance')
    plt.legend()
    if xtick != None:
        ax.set_xticklabels([0,1,10,20,30,40,50,60,70, 80])
        
def loss_split(x):
    if len(x) != 5:
        raise TypeError('Your input length is not 5!')
    
    upper = []
    lower = []
    avg = []
    for i1, i2, i3, i4, i5 in zip(x[0], x[1], x[2], x[3], x[4]):
        upper.append(max([i1, i2, i3, i4, i5]))
        lower.append(min([i1, i2, i3, i4, i5]))
        avg.append(np.median([i1, i2, i3, i4, i5]))
    return avg ,upper, lower


def fft_2d(x):
    fft_x = np.fft.fft2(x)
    fft_x = np.fft.fftshift(fft_x)
    mag = np.abs(fft_x)
    dB = 20 * np.log10(mag)
    return dB

def fft_1d(x):
    fft_x = np.fft.fft(x)
    fft_x = np.fft.fftshift(fft_x)
    mag = np.abs(fft_x)
    dB = 20 * np.log10(mag)
    
    ph = np.arctan(fft_x.imag / fft_x.real)
    deg = ph * 180/np.pi
    return dB, deg

def plot_2D_colorbar(x, t = 'title', xl = 'xlabel', yl = 'ylabel'):
    plt.figure()
    plt.title(t)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.imshow(x)
    plt.colorbar()

def plot_1D_colorbar(mag, ph):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12, 6))

    axes[0].plot(mag)
    axes[0].set_title('Magnitude (dB)')
    axes[0].set_xlabel('subcarriers')
    axes[0].set_ylabel('Mag (dB)')

    axes[1].plot(ph)
    axes[1].set_title('Phase (ph)')
    axes[1].set_xlabel('subcarriers')
    axes[1].set_ylabel('Ph (deg)')

    plt.tight_layout()

def signal_processing_2D(x, sample = None, antenna = None):
    if x.ndim != 2:
        raise TypeError('Wrong input dimension!')
    
    if sample == None:
        sample = np.random.randint(0, x.shape[0])
    
    data = x[sample]
    data = data.reshape(64, 100)
    '2D magnitude'
    plot_2D_colorbar(20 * np.log10(np.abs(data)), t = '2D magnitude (dB)', xl = 'subcarriers', yl = 'antennas')
    
    '2D FFT magnitude'
    mag = fft_2d(data)
    plot_2D_colorbar(mag, t = '2D FFT magnitude (dB)', xl = 'subcarriers', yl = 'antennas')
    
    '1D FFT analysis of 1 antenna'
    if antenna == None:
        antenna = np.random.randint(0, data.shape[0])
    m, p = fft_1d(data[antenna])
    plot_1D_colorbar(m, p)