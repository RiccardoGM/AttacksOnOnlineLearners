## Import statements ##
import matplotlib.pyplot as plt

''' Plotting style set-up '''

def SetPlotParams(magnification=1.0, ratio=float(2.2/2.7), fontsize=11., ylabelsize=None, xlabelsize=None, lines_w=1.5, ms=1.2):

    plt.style.use('ggplot')

    if (ylabelsize==None):
        ylabelsize = fontsize
    if (xlabelsize==None):
        xlabelsize = fontsize

    ratio = ratio  # usually this is 2.2/2.7
    fig_width = 2.9 * magnification # width in inches
    fig_height = fig_width*ratio  # height in inches
    fig_size = [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True

    plt.rcParams['lines.linewidth'] = lines_w
    #plt.rcParams['lines.markeredgewidth'] = 0.25
    #plt.rcParams['lines.markersize'] = 1
    plt.rcParams['lines.markeredgewidth'] = 1.
    plt.rcParams['errorbar.capsize'] = 1 #1.5

    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.markerscale'] = 1
    plt.rcParams['legend.handlelength'] = 1.
    plt.rcParams['legend.labelspacing'] = 0.3
    plt.rcParams['legend.columnspacing'] = 0.3
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0.0'
    plt.rcParams['axes.linewidth'] = '0.7'
    plt.rcParams['lines.markersize'] = ms

    plt.rcParams['grid.color'] = '0.85'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = '0.7'
    plt.rcParams['grid.alpha'] = '1.'

    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['xtick.labelsize'] = xlabelsize
    plt.rcParams['ytick.labelsize'] = ylabelsize
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'

    plt.rcParams['xtick.major.size'] = 3.
    plt.rcParams['xtick.major.width'] = 0.7
    plt.rcParams['xtick.minor.size'] = 0
    plt.rcParams['ytick.major.size'] = 3.
    plt.rcParams['ytick.major.width'] = 0.7
    plt.rcParams['ytick.minor.size'] = 0
    plt.rcParams['xtick.major.pad']= 5.
    plt.rcParams['ytick.major.pad']= 5.
