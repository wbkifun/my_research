#!/bin/sh

mencoder mf://*.png -mf fps=5:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o output.avi
