# https://stackoverflow.com/a/47749903

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.yticks([])
plt.xticks([])
plt.rcParams["font.family"] = 'serif'

colors = ["blue", "red", "green", "black"]
f = lambda m, c: plt.plot([], [], marker=m, color=c, linewidth=4, linestyle='--')[0]
handles = [f("_", colors[i]) for i in range(2)]
# handles.append(f("_", colors[-1]))
labels = ["Monitored MBIE-EB", "Directed-E$\mathbf{^2}$", "Minimax Optimal"]
handles.append(plt.axhline(5, linestyle="dashed", color="k", linewidth=3, label="Optimal"))
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True, ncols=3, prop={'weight':'bold'})


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(f"/Users/alirezakazemipour/Desktop/legend.pdf",
                format="pdf",
                bbox_inches=bbox
                )


export_legend(legend)
# plt.show()