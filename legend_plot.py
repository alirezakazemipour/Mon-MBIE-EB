# https://stackoverflow.com/a/47749903

import matplotlib.pyplot as plt

plt.yticks([])
plt.xticks([])
plt.rcParams["font.family"] = 'sans-serif'

my_color = "#2a9d8f"
simone_color = "#f4a261"
known_color = "#C26DBC"

colors = [my_color, simone_color, known_color, "black"]
f = lambda m, c: plt.plot([], [], marker=m, color=c, linewidth=4, linestyle='--')[0]
handles = [f("_", colors[i]) for i in range(1)]
# handles.append(f("_", colors[-1]))
labels = ["Monitored MBIE-EB", "Directed-E$^2$", "Known Monitor", "Minimax-Optimal"]
handles.append(plt.axhline(6, linestyle=":", color=simone_color, linewidth=3, label="Optimal"))
handles.append(plt.axhline(7, marker="o", color=known_color, linewidth=3))
handles.append(plt.axhline(5, linestyle="dashed", color="black", linewidth=3))
legend = plt.legend(handles, labels, loc=(1, 1), framealpha=1, frameon=False, ncols=5)


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