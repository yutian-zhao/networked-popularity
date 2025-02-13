import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt

class ColorPalette:
    # cornflower_blue
    BLUE = '#6495ed'
    # tomato
    TOMATO = '#ff6347'
    # color_cycle_4
    CC4 = ['#4486F4', '#1CA45C', '#FF9E0F', '#DA483B']
    # color_cycle_4 in RGB format
    CCRGB4 = [(68, 134, 244), (28, 164, 92), (255, 158, 15), (218, 72, 59)]
    # color_cycle_6
    CC6 = ['#255FDB', '#4285F4', '#91BFFF', '#C4E1FF', '#FF9E0F', '#DA483B']
    # color_cycle_8
    CC8 = ['#EC5f67', '#F99157', '#FAC863', '#99C794', '#5FB3B3', '#6699CC', '#C594C5', '#AB7967']
    # label fontsize
    LABELFS = 11
    # title fontsize
    TITLEFS = 12
    # tick style
    TICKSTYLE = {'axis': 'both', 'labelsize': 10}
    # bar text style
    BARTEXTSTYLE = {'ha': 'center', 'va': 'bottom', 'color': 'k', 'fontsize': 10}


def concise_fmt(x, pos):
    if abs(x) // 10000000 > 0:
        return '{0:.0f}M'.format(x / 1000000)
    elif abs(x) // 1000000 > 0:
        return '{0:.1f}M'.format(x / 1000000)
    elif abs(x) // 10000 > 0:
        return '{0:.0f}K'.format(x / 1000)
    elif abs(x) // 1000 > 0:
        return '{0:.0f}K'.format(x / 1000)
    else:
        return '{0:.0f}'.format(x)


def hide_spines(axes):
    if isinstance(axes, Iterable):
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    else:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)


def stackedBarPlot(ax,  # axes to plot onto
                   data,  # data to plot
                   cols,  # colors for each level
                   # xLabels=None,  # bar specific labels
                   yTicks=6.,
                   # information used for making y ticks ["none", <int> or [[tick_pos1, tick_pos2, ... ],[tick_label_1, tick_label2, ...]]
                   edgeCols=None,  # colors for edges
                   showFirst=-1,  # only plot the first <showFirst> bars
                   scale=False,  # scale bars to same height
                   widths=None,  # set widths for each bar
                   heights=None,  # set heights for each bar
                   ylabel='',  # label for x axis
                   xlabel='',  # label for y axis
                   alpha=0.6,  # alpha of bar
                   gap=0.,  # gap between bars
                   endGaps=False  # allow gaps at end of bar chart (only used if gaps != 0.)
                   ):
    # ------------------------------------------------------------------------------
    # make sure this makes sense
    if showFirst != -1:
        showFirst = np.min([showFirst, np.shape(data)[0]])
        data_copy = np.copy(data[:showFirst]).transpose().astype('float')
        data_shape = np.shape(data_copy)
        if heights != None:
            heights = heights[:showFirst]
        if widths != None:
            widths = widths[:showFirst]
        showFirst = -1
    else:
        data_copy = np.copy(data).transpose()
    data_shape = np.shape(data_copy)

    # determine the number of bars and corresponding levels from the shape of the data
    num_bars = data_shape[1]
    levels = data_shape[0]

    if widths is None:
        widths = np.array([1] * num_bars)
        x = np.arange(num_bars)
    else:
        x = [0]
        for i in range(1, len(widths)):
            x.append(x[i - 1] + (widths[i - 1] + widths[i]) / 2)

    # stack the data --
    # replace the value in each level by the cumulative sum of all preceding levels
    data_stack = np.reshape([float(i) for i in np.ravel(np.cumsum(data_copy, axis=0))], data_shape)

    # scale the data is needed
    if scale:
        data_copy /= data_stack[levels - 1]
        data_stack /= data_stack[levels - 1]
        if heights != None:
            print("WARNING: setting scale and heights does not make sense.")
            heights = None
    elif heights != None:
        data_copy /= data_stack[levels - 1]
        data_stack /= data_stack[levels - 1]
        for i in np.arange(num_bars):
            data_copy[:, i] *= heights[i]
            data_stack[:, i] *= heights[i]

    # ------------------------------------------------------------------------------
    # ticks

    if yTicks != "none":
        # it is either a set of ticks or the number of auto ticks to make
        real_ticks = True
        try:
            k = len(yTicks[1])
        except:
            real_ticks = False

        if not real_ticks:
            yTicks = float(yTicks)
            if scale:
                # make the ticks line up to 100 %
                y_ticks_at = np.arange(yTicks) / (yTicks - 1)
                y_tick_labels = np.array(["%0.2f" % (i * 100) for i in y_ticks_at])
            else:
                # space the ticks along the y axis
                y_ticks_at = np.arange(yTicks) / (yTicks - 1) * np.max(data_stack)
                y_tick_labels = np.array([str(i) for i in y_ticks_at])
            yTicks = (y_ticks_at, y_tick_labels)

    # ------------------------------------------------------------------------------
    # plot

    if edgeCols is None:
        edgeCols = ["none"] * len(cols)

    # take cae of gaps
    gapd_widths = [i - gap for i in widths]

    # bars
    ax.bar(x,
           data_stack[0],
           color=cols[0],
           edgecolor=edgeCols[0],
           width=gapd_widths,
           linewidth=0.6,
           align='center'
           )

    for i in np.arange(1, levels):
        ax.bar(x,
               data_copy[i],
               bottom=data_stack[i - 1],
               color=cols[i],
               edgecolor=edgeCols[i],
               width=gapd_widths,
               linewidth=0.5,
               alpha=alpha,
               align='center'
               )

    # make ticks if necessary
    if yTicks != "none":
        ax.tick_params(axis='y', which='both', direction="out", labelsize=10)
        ax.yaxis.tick_left()
        # ax.yticks(yTicks[0], yTicks[1])
    # else:
        # ax.yticks([], [])

    ax.tick_params(axis='x', which='both', direction="out", labelsize=10)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_ticklabels(['{0:.0f}'.format(xx+1) for xx in ax.get_xticks().tolist()])

    # limits
    if endGaps:
        ax.set_xlim(-1. * widths[0] / 2. - gap / 2., np.sum(widths) - widths[0] / 2. + gap / 2.)
    else:
        ax.set_xlim(-1. * widths[0] / 2. + gap / 2., np.sum(widths) - widths[0] / 2. - gap / 2.)
    ax.set_ylim(0, yTicks[0][-1])  # np.max(data_stack))

    # labels
    if xlabel != '':
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel != '':
        ax.set_ylabel(ylabel, fontsize=11)

## scatter plot with a mean point plotted for each x value
def mean_scatter(x, y, c1='grey', c2='blue'):
    plt.scatter(x, y, s=2, color=c1, marker='o', alpha=0.5)
    data = np.zeros((len(x), 2))
    data[:, 0] = x
    data[:, 1] = y
    data = data[data[:, 0].argsort()]
    data_split = np.split(data[:,1], np.unique(data[:, 0], return_index=True)[1][1:])
    y_mean = [np.mean(i) for i in data_split]
    plt.scatter(np.sort(np.unique(data[:, 0])), y_mean, s=2, color=c2, marker='o', alpha=1)

## scatter plot with a average line plotted
def mean_plot(x, y, c1='grey', c2='blue'):
    plt.scatter(x, y, s=2, color=c1, marker='o', alpha=0.5)
    data = np.zeros((len(x), 2))
    data[:, 0] = x
    data[:, 1] = y
    data = data[data[:, 0].argsort()]
    data_split = np.split(data[:,1], np.unique(data[:, 0], return_index=True)[1][1:])
    y_mean = [np.mean(i) for i in data_split]
    plt.plot(np.sort(np.unique(data[:, 0])), y_mean, color=c2)

## Cumulative percentage of views vs (inverse) view percentile
def plot_cumulative(avg_view_list):
    avg_view_sort_ind = np.argsort(avg_view_list)
    sorted_avg_view_list = avg_view_list[avg_view_sort_ind]
    # sorted_ids = np.arange(num_videos)[avg_view_sort_ind]
    total_views = np.sum(avg_view_list)


    # calculate head_mid and mid_tail points in the long tail model
    # N_50 = num_videos - np.argmin(np.abs([(100*np.sum(sorted_avg_view_list[i:])/total_views-50) for i in range(num_videos)]))
    # head_mid = math.floor(np.power(N_50, 2/3))
    # mid_tail = math.ceil(np.power(N_50, 4/3))
    # head_mid_pho = np.power(N_50, 2/3)/num_videos
    # mid_tail_pho = np.power(N_50, 4/3)/num_videos
    # print("head_mid_pho, N_50/num_videos, mid_tail_pho: ", head_mid_pho, N_50/num_videos, mid_tail_pho)
    # print("N_50, head_mid, mid_tail: ", N_50, head_mid, mid_tail)
    # head_ids = sorted_ids[-head_mid:]
    # mid_ids = sorted_ids[mid_tail+1:head_mid]
    # tail_ids = sorted_ids[:mid_tail]

    percentiles = np.arange(1000)/10
    percentages = np.array([100*np.sum(avg_view_list[avg_view_list>=np.percentile(avg_view_list, 100-p)])/total_views for p in percentiles])
    # print("gini: ", gini(avg_view_list))
    
#     plt.vlines([head_mid_pho, N_50/num_videos, mid_tail_pho], 0, 100, linestyle='dashed')
    plt.plot(percentiles, percentages)