# https://stackoverflow.com/a/47749903

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.yticks([])
plt.xticks([])
plt.rcParams["font.family"] = 'serif'

colors = ["blue", "red"]
f = lambda m, c: plt.plot([], [], marker=m, color=c, linewidth=4)[0]
handles = [f("_", colors[i]) for i in range(2)]
labels = ["Double MBIE", "Parisi et al's"]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False, ncols=2)


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(f"/Users/alirezakazemipour/Desktop/legend.pdf",
                format="pdf",
                bbox_inches=bbox
                )


export_legend(legend)
plt.show()
