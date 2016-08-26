##=============================================================================#
## R code to draw figures accompany acompanying the poster example.
## All the outputs will be stored in the "figs" folder. 
##
## Author: Daina Chiba, Rice University (daina.chiba@gmail.com)
## Last modified on: April 29, 2012
##=============================================================================#
rm(list=ls())    # Clear memory. 

##==================================================#
## Figure in the left column (endogeneity)
##==================================================#
require(diagram)
pdf(file="figs/figEndog.pdf",width=8, height=4)
par(mar = c(0, 0, 0, 0))
openplotmat()
elpos <- coordinates(c(1,2))
arrpos <- matrix(ncol=2,nrow=3)
arrpos[1,] <- straightarrow(to = elpos[3,], from = elpos[2,], lwd = 3, arr.pos = .5, arr.length = 1)
arrpos[2,] <-straightarrow(to = elpos[2,], from = elpos[1,], lwd = 3, arr.pos = .7, arr.length = 1,lty=2)
arrpos[3,] <-straightarrow(to = elpos[3,], from = elpos[1,], lwd = 3, arr.pos = .7, arr.length = 1,lty=2)
textround(elpos[1,], 0.11,0.1, lab = "Something that\ncauses X and Y", box.col = "gray90",
 shadow.col = "black", shadow.size = 0.005, cex = 1.5)
textrect (elpos[2,], 0.11, 0.1,lab = "Independent\nvariable (X)", box.col = "white",
 shadow.col = "black", shadow.size = 0.005, cex = 1.5)
textrect (elpos[3,], 0.13, 0.1,lab = "Dependent\nvariable (Y)", box.col = "white",
 shadow.col = "black", shadow.size = 0.005, cex = 1.5)
text(arrpos[2, 1] - 0, arrpos[2, 2] + .15, "+",cex=3)
text(arrpos[3, 1] - 0, arrpos[2, 2] + .15, "-", cex=3)
text(arrpos[1, 1], arrpos[1, 2] - .07, "+ ?", cex=3)
dev.off()

##==================================================#
## Kernel density plots in the middle column
##==================================================#
## Figures and codes here are copied from Hadley Wickham's webpage (http://had.co.nz/ggplot2/). 
## Hadley Wickham is the author of the ggplot2 package.

require(ggplot2)
### left one
g <- ggplot(diamonds, aes(price, fill = cut)) + geom_density(alpha = 0.2)
g + opts(plot.background=theme_rect(fill="transparent"), panel.background=theme_rect(fill="transparent")) + theme_bw()
ggsave(file="figs/figLeft.pdf",width=8,height=6)

### right one
g <- ggplot(diamonds, aes(depth, fill = cut)) + geom_density(alpha = 0.2) + xlim(55, 70)
g + opts(plot.background=theme_rect(fill="transparent"), panel.background=theme_rect(fill="transparent")) + theme_bw()
ggsave(file="figs/figRight.pdf",width=8,height=6)

##==================================================#
## "Results" figure
##==================================================#
g <- ggplot(mtcars, aes(x = wt, y=mpg), . ~ cyl) + geom_point() + 
  opts(plot.background=theme_rect(fill="transparent"), panel.background=theme_rect(fill="transparent")) + theme_bw()
df <- data.frame(a=rnorm(10, 25), b=rnorm(10, 0)) 
coefs <- ddply(mtcars, .(cyl), function(df) {
  m <- lm(mpg ~ wt, data=df)
  data.frame(a = coef(m)[1], b = coef(m)[2]) 
  })    
str(coefs)
g <- g + geom_smooth(aes(group=cyl), method="lm") 
print(g)
ggsave(file="figs/figResult.pdf",width=8,height=6)
