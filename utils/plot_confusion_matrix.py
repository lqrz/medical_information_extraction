
from ggplot import *
import pandas as pd
from custom_geom_tile import custom_geom_tile
import matplotlib.pyplot as plt
import os
import six
import sys

# def plot_confusion_matrix(confusion_matrix, labels, output_filename, title=None):
#     df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
#     df['true'] = df.index
#     df_melted = pd.melt(df, id_vars=['true'], var_name='prediction')
#     plot = ggplot(aes(x='prediction', y='true'), df_melted) + \
#         custom_geom_tile(aes(fill='value')) + \
#         labs(x='Predicted labels', y='True labels') + \
#         scale_color_gradient()
#
#     if title:
#         plot += ggtitle(title)
#
#     # plot.__repr__()
#
#     #use bbox_inches='tight' for matplotlib to remove margins in image.
#     ggsave_lqrz(output_filename, plot, dpi=100, bbox_inches='tight')

def plot_confusion_matrix_ggplot(confusion_matrix, labels, output_filename, title=None):
    import rpy2.robjects as robj
    import rpy2.robjects.pandas2ri  # for dataframe conversion
    from rpy2.robjects.packages import importr

    df = pd.DataFrame(confusion_matrix[::-1], index=labels[::-1], columns=labels)
    df['true'] = df.index
    df_melted = pd.melt(df, id_vars=['true'], var_name='prediction')

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df_melted)
    plotFunc = robj.r("""
        library(ggplot2)

        function(df, output_filename){
            #df$sentence_nr <- as.character(df$sentence_nr)
            df$true  <- as.character(df$true)
            df$true  <- factor(df$true, levels=unique(df$true))
            df$prediction  <- as.character(df$prediction)
            df$prediction  <- factor(df$prediction, levels=unique(df$prediction))
            str(df)

            p <- ggplot(df, aes(x=prediction, y=true, fill=value)) +
                geom_tile(colour='gray92') +
                # scale_fill_gradient(low='gray99', high='steelblue4', guide = guide_legend(title = "Probability")) +
                scale_fill_gradient(low='white', high='steelblue', guide = guide_colourbar(title="", ticks=FALSE,
                                                                                barwidth = 0.5, barheight = 12)) +
                # scale_fill_gradient() +
                labs(x='Predicted labels', y='True labels', title='') +
                theme(
                    panel.grid.major = element_blank(),
                    panel.border = element_blank(),
                    panel.background = element_blank(),
                    axis.ticks = element_blank(),
                    axis.text.x = element_text(angle=90, hjust=1, vjust=0.5))
                    #legend.position="none")

            print(p)

            ggsave(output_filename, plot=p, height=9, width=10, dpi=120)

            }
        """)

    plotFunc(conv_df, output_filename)
    gr.dev_off()

    return True

def ggsave_lqrz(filename=None, plot=None, device=None, format=None,
           path=None, scale=1, width=None, height=None, units="in",
           dpi=300, limitsize=True, **kwargs):
    """Save a ggplot with sensible defaults

    ggsave is a convenient function for saving a plot.  It defaults to
    saving the last plot that you displayed, and for a default size uses
    the size of the current graphics device.  It also guesses the type of
    graphics device from the extension.  This means the only argument you
    need to supply is the filename.

    Parameters
    ----------
    filename : str or file
        file name or file to write the plot to
    plot : ggplot
        plot to save, defaults to last plot displayed
    format : str
        image format to use, automatically extract from
        file name extension
    path : str
        path to save plot to (if you just want to set path and
        not filename)
    scale : number
        scaling factor
    width : number
        width (defaults to the width of current plotting window)
    height : number
        height (defaults to the height of current plotting window)
    units : str
        units for width and height when either one is explicitly
        specified (in, cm, or mm)
    dpi : number
        dpi to use for raster graphics
    limitsize : bool
        when `True` (the default), ggsave will not save images
        larger than 50x50 inches, to prevent the common error
        of specifying dimensions in pixels.
    kwargs : dict
        additional arguments to pass to matplotlib `savefig()`

    Returns
    -------
    None

    Examples
    --------
    # >>> from ggplot import *
    # >>> gg = ggplot(aes(x='wt',y='mpg',label='name'),data=mtcars) + geom_text()
    # >>> ggsave("filename.png", gg)

    Notes
    -----
    Incompatibilities to ggplot2:

    - `format` can be use as a alternative to `device`
    - ggsave will happily save matplotlib plots, if that was the last plot
    """


    fig_kwargs = {}
    fig_kwargs.update(kwargs)

    # This is the case when we just use "ggsave(plot)"
    if hasattr(filename, "draw"):
        plot, filename = filename, plot

    if plot is None:
        figure = plt.gcf()
    else:
        if hasattr(plot, "draw"):
            figure = plot.draw()
        else:
            raise Exception("plot is not a ggplot object")

    if format and device:
        raise Exception("Both 'format' and 'device' given: only use one")
    # in the end the imageformat is in format
    if device:
        format = device
    if format:
        if not format in figure.canvas.get_supported_filetypes():
            raise Exception("Unknown format: {0}".format(format))
        fig_kwargs["format"] = format

    if filename is None:
        if plot:
            # ggplot2 defaults to pdf
            filename = str(plot.__hash__()) + "." + (format if format else "pdf")
        else:
            # ggplot2 has a way to get to the last plot, but we currently dont't
            raise Exception("No filename given: please supply a filename")

    if not isinstance(filename, six.string_types):
        # so probably a file object
        if format is None:
            raise Exception("filename is not a string and no format given: please supply a format!")

    if path:
        filename = os.path.join(path, filename)

    if units not in ["in", "cm", "mm"]:
        raise Exception("units not 'in', 'cm', or 'mm'")

    to_inch = {"in": lambda x: x, "cm": lambda x: x / 2.54, "mm": lambda x: x * 2.54 * 10}
    from_inch = {"in": lambda x: x, "cm": lambda x: x * 2.54, "mm": lambda x: x * 2.54 * 10}

    w, h = figure.get_size_inches()
    issue_size = False
    if width is None:
        width = w
        issue_size = True
    else:
        width = to_inch[units](width)
    if height is None:
        height = h
        issue_size = True
    else:
        height = to_inch[units](height)

    try:
        scale = float(scale)
    except:
        raise Exception("Can't convert scale argument to a number: {0}".format(scale))
    # ggplot2: if you specify a width *and* a scale, you get the width*scale image!
    width = width * scale
    height = height * scale

    if issue_size:
        msg = "Saving {0} x {1} {2} image.\n".format(from_inch[units](width), from_inch[units](height), units)
        sys.stderr.write(msg)

    if limitsize and (width > 25 or height > 25):
        msg = "Dimensions exceed 25 inches (height and width are specified in inches/cm/mm, not pixels)." + \
              " If you are sure you want these dimensions, use 'limitsize=False'."
        raise Exception(msg)

    fig_kwargs["dpi"] = dpi

    # savefig(fname, dpi=None, facecolor='w', edgecolor='w',
    #    orientation='portrait', papertype=None, format=None,
    #    transparent=False, bbox_inches=None, pad_inches=0.1,
    #    frameon=None)
    try:
        figure.set_size_inches(width, height)

        for i, ax in enumerate(plt.gcf().axes):
            if i != 0:
                ax.xaxis.label.set_text('')
                ax.set_ylabel('')

        figure.savefig(filename, **fig_kwargs)
    finally:
        # restore the sizes
        figure.set_size_inches(w, h)
    # close figure, if it was drawn by ggsave
    if not plot is None:
        plt.close(figure)