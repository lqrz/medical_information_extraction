__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import cPickle
import pandas as pd

from data import get_testing_classification_report_labels
from trained_models import get_tf_rnn_path

def plot_confusion_matrix(confusion_matrix, labels, output_filename, title=None):
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
                scale_fill_gradient2(low='red4', high='green4', midpoint=0, mid = "white", guide = guide_colourbar(title="", ticks=FALSE,
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

def determine_output_filename(output_model):
    get_output_path = None

    if output_model == 'rnn':
        get_output_path = get_tf_rnn_path

    assert get_output_path is not None

    return get_output_path('cm_difference.png')

if __name__ == '__main__':

    assert sys.argv.__len__() == 4

    cm1_path = sys.argv[1]
    cm2_path = sys.argv[2]
    output_model = sys.argv[3]

    output_filename = determine_output_filename(output_model)

    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'

    cm1 = cPickle.load(open(root + cm1_path, 'rb'))
    cm2 = cPickle.load(open(root + cm2_path, 'rb'))

    assert cm1.shape == cm2.shape

    cm_dif = cm1 - cm2

    labels = test_labels_list = get_testing_classification_report_labels()

    plot_confusion_matrix(cm_dif, labels, output_filename)