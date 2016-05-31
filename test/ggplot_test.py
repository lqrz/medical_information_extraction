__author__ = 'lqrz'
from ggplot import *
import pandas as pd

if __name__ == '__main__':

    df = pd.DataFrame({"x": [1, 2, 3, 4], "value": [1, 3, 4, 2]})
    p = ggplot(aes(x="x", weight="value"), data=df) + \
        geom_bar()

    gg = ggplot(diamonds, aes(x='clarity', fill='cut', color='cut')) + \
         stat_bin(binwidth=1200)

    ggsave(plot=gg, filename='ggplot_test.png', dpi=100)
