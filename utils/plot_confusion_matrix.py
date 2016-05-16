
from ggplot import *
import pandas as pd
from custom_geom_tile import custom_geom_tile

def plot_confusion_matrix(confusion_matrix, labels, output_filename, title=None):
    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    df['true'] = df.index
    df_melted = pd.melt(df, id_vars=['true'], var_name='prediction')
    plot = ggplot(aes(x='prediction', y='true'), df_melted) + \
        custom_geom_tile(aes(fill='value')) + \
        labs(x='Predicted labels', y='True labels') + \
        scale_color_gradient()

    if title:
        plot += ggtitle(title)
    # plot.__repr__()   #builds the plot

    #use bbox_inches='tight' for matplotlib to remove margins in image.
    ggsave(output_filename, plot, dpi=100, bbox_inches='tight')