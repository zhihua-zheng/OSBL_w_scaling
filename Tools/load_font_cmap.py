import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


font_dirs = '../Data/OpenSans'
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# set font
# plt.rcParams['font.family'] = 'Open Sans'

# add customized colormap
xkcd_colors = ['xkcd:'+i for i in ['heather', 'pale blue', 'azure', 'sapphire']]
hexstr = [mcolors.XKCD_COLORS[i] for i in xkcd_colors]
rgba = mcolors.to_rgba_array(hexstr)
nodes = [0.0, 0.1, 0.5, 1.0]
cmap_count = LinearSegmentedColormap.from_list('nl_count', list(zip(nodes, rgba)))
# mpl.cm.unregister_cmap('nl_count')
mpl.colormaps.register(cmap=cmap_count)