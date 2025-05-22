# Image Desmosinator

This project was created because I was too lazy to trace Shrek's swamp in Desmos, but I wasn't too lazy to write a Python script to do it for me.

## How to use this
1. Download an image you wanna Desmosinate and put that image in the `images` folder.
2. Open up `desmosinator.py` and, in the `main()` function, you'll see the variables `source_image` and `output_name`. Set `source_image` to the name of the file that you want to Desmosinate, and set `output_name` to what you want to call your output.
3. Run `desmosinator.py`. It may take a lil while.
4. Once the script finishes running, go into the `outputs` folder and find your output file. Copy and paste all the equations into Desmos, and there you go!

NOTE: if your output is more than around 5000 equations long, don't expect Desmos to be able to handle it.

## How does this work?
First, this script uses the Sobel operator to detect edges in the source image. It then cleans it up to (attempt to) reduce the noise in the image. Next, it uses the [PyPotrace](https://github.com/flupke/pypotrace) library to convert the bitmap image to a vector graphic composed of Bézier curves and corners. Finally, the script spits out the equations of those Bézier curves and corners into a Desmos-copy-pasteable form.

## Credit
Credit to kevinjycui's [DesmosBezierRenderer](https://github.com/kevinjycui/DesmosBezierRenderer?tab=readme-ov-file) for the original idea and a (frankly) much better execution of it. This is how I stumbled upon Potrace.

Credit to [this Desmos graph](https://www.desmos.com/calculator/cahqdxeshd) (I don't know who made it. Maybe the Desmos team) for the general equation of a cubic Bézier curve.